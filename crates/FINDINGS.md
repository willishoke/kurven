# Web port findings

What the gate turned up in either implementation, every tolerance chosen and
why, and every place the port differs from Swift. One dated entry each, newest
last. (HANDOFF-web-port.md names `docs/handoff/web-port-findings.md`; that
directory is kept out of git, so the file lives here.)

## 2026-11-15 · Where things are, and two additions to the plan

The Cargo workspace's root is `crates/Cargo.toml`, so everything Rust is under
`crates/` and its build tree is `crates/target/`. The crates are the handoff's
(`kurven-math`, `kurven-core`, `kurven-landscape`, `kurven-surfaces`,
`kurven-bake`, `kurven-session`, `kurven-wasm`, `kurven-cli`) plus one:
`kurven-golden`, the port's golden sets. They are written by native
`kurven-cli fixtures` and by the wasm module under Node, and the two must agree
byte for byte, so the generator has to live in a library both depend on. In
Swift it is part of `kurven-cli` (`Fixtures.swift`).

`kurven-wasm` exports the plain C ABI (pointer and length), and
`crates/kurven-wasm/goldens.mjs` reads it with `WebAssembly.instantiate` and
Node's standard library. That keeps the harness free of generated glue, of the
wasm-bindgen CLI (whose version has to match the crate's), and of npm packages.
Whether the page's binding is this or wasm-bindgen is a Phase 6 decision, made
with the Worker protocol.

## 2026-11-15 · Darwin's libm and the libm crate disagree on about one call in ten

I measured this on this machine (aarch64 macOS) with a scratch program outside
the repository: Rust's std (which calls Darwin's libm here) against the libm
crate, on 220,001 arguments in [−50, 50] and (0, 0.02] for each of fourteen
functions, plus 160,000 pairs each for atan2, pow and hypot. 315,263 of
3,262,014 results differ, all by 1 to 3 ulp: tan in 40% of calls (up to
3 ulp), sinh 18%, cosh 14%, exp 9%, sin and cos 4%, log10 0.3%. Swift calls
Darwin's libm, so this is how far apart the port and the reference start. It is
why drawings are compared to a tolerance and not as bytes, and why a bisection
that falls the other way at a tangency is an acceptable count difference.

## 2026-11-15 · The libm crate's own fused multiply-add, and NaN bits

`libm` 0.2.16's `cbrt` calls `fma` internally, and its default `arch` feature
uses hardware instructions for operations that are correctly rounded anyway
(sqrt, floor, ceil, trunc, rint, fma). A correctly rounded fma gives the same
answer in hardware and in software, so this breaks neither native and wasm
agreeing nor the rule. The rule ("no fused multiply-add") is about the port's
own `a * b + c`. Rust never contracts that into a fused one. clippy refuses
`mul_add`, and the libm table has three `unfused` lines whose answers would
change if a target contracted them. The only arch-specific transcendental code
in the crate is an x87 path for x86 without SSE, which is not a target.

IEEE 754 leaves the bits of a computed NaN to the platform, and wasm's are
nondeterministic. So no golden file may write a NaN's bits: the libm table
writes `nan`, and later goldens must do the same. Canonical JSON cannot hold a
NaN in any case.

## 2026-11-15 · The libm grep cannot tell types apart

`scripts/check-web.sh` refuses `.sin(`, `.exp(`, `.ln(`, `.log(`, `.powf(`,
`.atan2(`, `.hypot(`, `.cbrt(`, `.mul_add(` and the rest, and `f64::sin` and
its kin, anywhere in `crates/` except `kurven-math/src/real.rs`. A method of
the port's own with one of those names, such as `Complex::exp` called as
`z.exp()`, would trip the grep too, although clippy (which knows the types)
would let it through. Phase 1 has to decide how to handle this before it ports
complex arithmetic. I recommend free functions in a `complex` module
(`complex::exp(z)`), which read like Swift's free `exp(z)` overloads and keep
the grep as it is. The other option is to drop the grep and rely on clippy
alone.

## 2026-11-15 · The reference drawings

`kurven-cli references tests/fixtures DIR` is the one Swift change, an additive
subcommand (commit 864b81b). Its output is committed under
`crates/fixtures/swift/`: 25 SVGs and `index.json`, 6,268,758 bytes in all (the
largest is gallery-cn.svg at 1.2 MB). That is under the 20 MB limit, so the
drawings are kept as Swift writes them. The list:

- the gallery as `docs/gallery/make.py` sets it up: gamma, cn, the four cubic
  corners, the frame's edges and its Turing hatching
- the native landscapes of `tests/fixtures/expr`: gamma, cubic, typed, tower.
  There are four, not the three the handoff and `docs/development.md` count.
- sn flat, and sn in relief at 0.4
- a torus with a 2/5 linear curve
- `periodic` with sn(z, 0.5) over its periods
- the slope-lines golden of PR #60
- the five Platonic solids, Császár's torus and the picture frame
- the cube-and-torus composite

Every drawing was baked. Each is drawn the way its command draws it, and I
checked that byte for byte where there is something to check against:

- the six gallery landscapes against make.py's raw bakes (before its thinning
  and scaling)
- slope-lines against `tests/fixtures/fields/landscape_slope_lines.svg`
- composite-three against `tests/fixtures/composite/three.svg`
- sn and the Turing frame against `surface ... -o`

Two runs write the same bytes. `scripts/check.sh` stayed green with the
subcommand in (2017 checks). The drawings were baked on Apple Silicon; the bake
uses Metal for its depth buffer, but no visibility verdict comes from it.

## 2026-11-15 · What the SVGs can and cannot show

- **Precision floor.** SVG.swift writes coordinates to four decimals, so each
  is off by up to 5e-5 page units and a point by up to 7.1e-5. A distance
  tolerance much below 1e-4 cannot be met against these files, whatever the
  port does.
- **Stroke widths.** Widths are written to six decimals of a page unit, scaled
  by the viewBox's width. `compare` therefore compares them in points, to
  1e-3 relative plus that rounding.
- **Empty layers.** SVG.swift leaves out an empty layer (`where
  !paths.isEmpty`). So `compare` matches groups by order, and a layer empty in
  one implementation and not the other shows up as a layer-count mismatch, not
  as a distance. `index.json` lists each drawing's non-empty layers by name.
  For example, gallery-gamma's `riser_hatch` is empty and absent.

## 2026-11-15 · compare's tolerances

The defaults are 1e-3 page units of Hausdorff distance (about fifteen times the
SVG rounding) and 1e-3 relative total length per layer. They are provisional:
no Rust drawing exists yet to test them against. Phases 3 to 5 set a tolerance
per drawing and record each one here, and none is loosened to get green. The
distance is computed to within a 64th of the tolerance, and the verdict is
taken on the upper end of that bound. Identical layers come out at 0 exactly.
The check confirms that every reference compared with itself is at 0, that a
drawing moved by half the tolerance passes with the distance it was moved, and
that a drawing moved by three times the tolerance, or with a stroke taken out,
fails.

## 2026-11-15 · Toolchain, CI, and what has not run

CI pins Rust 1.99.1 (the stable toolchain the goldens were written with) and
Node 26, on ubuntu-latest. The x86_64 Linux run has not been observed yet
because nothing has been pushed. Its first run is the test that the libm table
is the same bytes on a second platform; if it is not, that is a finding to
record here, not a golden to regenerate. The page pins typescript 7.0.2 (the
native compiler, through per-platform packages that the lock lists for Linux
too) and vite 8.3.5.

## 2026-11-15 · Small differences from Swift, by construction

- **JSON reader.** It refuses a key given twice; `JSONSerialization` keeps one
  silently. An integer literal past i64 is read as a Double. This is not
  checked against Swift.
- **npy writer.** Only the reader is ported so far. The writer comes with the
  bundle schema in Phase 2, with npy's round-trip suite.

## 2026-11-15 · The docs date the plates to 1909

`docs/glossary.md` ("Jahnke and Emde, *Funktionentafeln*, 1909", and the
definition of a plate) and `docs/architecture.md` ("drawn in 1909") date the
plates to 1909. The reliefs first appear in the 1933 second edition, as the
handoff says. I have not edited either file, since that is outside the port's
scope; the port's prose uses 1933.

## 2026-11-15 · Decided: complex functions are associated functions

The owner settled the question the grep raised. Complex functions are
associated functions, `Complex::exp(z)`, mirroring Swift's `Complex.exp(z)`
statics (`Sources/KurvenMath/Complex.swift`). Called that way they never
spell `.exp(`, so the grep stays as it is, and the Swift is findable from the
Rust by the same name in the same place.

## 2026-11-15 · Review of PR #63: what was fixed

- **compare, far points.** A point far off the other drawing's grid walked
  every ring of cells out to it, (distance / cell)² visits. The rings now
  start where the grid begins and are clamped to it. `nearest` always names a
  segment, and answers none for an empty set.
- **SVG reader.** `nan` and `inf` were read as coordinates. They are refused
  now, so the drawing cannot be read (exit 2). An empty `<polyline
  points=""/>`, which SVG.swift would write for a path of no vertices, is a
  stroke counted with no geometry. A multi-byte character at a tag's start is
  an error, not a panic.
- **npy reader.** Shape × item size, and the header's end, are checked for
  overflow; a shape like (2⁶², 4) used to wrap to zero bytes.
- **Golden gate.** It diffs against HEAD, not the index, and writes nothing to
  the index. `scripts/check.sh` has the same index-diff pattern for Swift's
  goldens (`git add --intent-to-add` then `git diff`, so a staged change to a
  golden passes). That script is out of the port's scope, so it is left as it
  is and recorded here.
- **Toolchain.** The toolchain is pinned in `crates/rust-toolchain.toml`
  (1.99.1, rustfmt, clippy, wasm32). check-web.sh runs cargo in `crates/` and
  installs the pin, and CI installs it the same way.
- **References.** The four cubic corners now come from one sampled bundle, as
  make.py does. The 25 drawings are byte-identical to the ones committed.

## 2026-11-15 · Phase 2: what came across, and what did not

Ported into `kurven-core`: Spaces, Camera, Navigation, Depth (DepthFrame,
DepthImage: a `Framing`'s raster is a `DepthFrame`), Grid, Polyline,
Bisection, Contour, JSON (the writer), NPY (the writer), SVG, Manifest whole
and SurfaceRequest whole. simd's vectors and matrices are `simd.rs` (Vec2/3/4,
Mat3/4, by columns, every product unfused in one order).

Ported ahead of their phase, because the schema decodes into them: the
construction half of Solid.swift (`Solid::checking`, `FaceFrame`, Euler
characteristic, shells, genus, convexity, edge paths) with all of
ConvexPartition.swift, since a polyhedron's manifest is checked by building
its solid; and the values of Hatching.swift (`Hatching`, `FaceSelector`,
`Tone`, `Turing`). The solid's occlusion, hatching, mesh, rigid motions and
named solids (convex hull, Platonic, frame) are left for Phase 5.

Left out, each for a later phase: **Bundle.swift** (`KurvenBundle` holds a
`Surface`, its walls a `Mesh`, its derived layers come from
`Surface.curves` and a `ContourRefine`: Phase 3; the schema it reads is
here), **ContourMemo.swift** (the refiner's memo, an extension of
`ContourRefine`: Phase 3), and the suite "contract: bundles decode into
consistent values", which builds every bundle's drawing
(`KurvenBundle.read`, `SurfaceBundle.read`, `PolyhedronBundle.read`:
Phases 3 to 5). In its place a test that every contract manifest decodes as
its kind (five landscapes, six surfaces, seven polyhedra).

Nothing needed Phase 1. A `periodic` surface keeps its expression as text,
as Swift's `SurfaceRequest` does (KurvenCore depends on no parser), and
round trips as text; what does not parse is refused when the drawing is
built (`SurfaceDrawing.build`, Phase 4).

## 2026-11-15 · Canonical numbers are Swift's, not Python's

`JSONValue.number` writes `Double.description` and says it agrees with
Python's `repr`. It does not, between 2^53 and 1e16: Swift writes
9007199254740994.0 as `9.007199254740994e+15`, Python as
`9007199254740994.0`. (The suite's "canonical JSON writes numbers the way
Python's json does" checks none of those.) The port follows Swift, which
wrote the golden files: positional from 1e-4 up to 2^53 inclusive,
`d.ddde±XX` outside, digits from Rust's shortest round-trip formatting. I
asked Swift for thirty-two values on this machine, the thresholds among
them, and the tests pin its answers; no contract fixture holds a number in
the disputed range. serde is not used: no stock serializer writes either
rule.

## 2026-11-15 · Phase 2 tolerances, all Swift's

- **camera/**: 1e-9 per component, Swift's. Measured worst: 7.1e-15
  (recip, elliptic), 3.6e-15 (zeta), over 264 points each; the cosines are
  the libm crate's here and Darwin's there.
- **contour/**: every segment midpoint within 1e-9 of one in contourpy's
  set, and the same path and segment counts, Swift's rule. All thirteen
  levels pass with no midpoint unmatched.
- **navigation**: Swift's, unchanged (1e-9, 1e-12, 1e-6 through a turn),
  and exact equality for "an orbit matching a preset IS the oblique
  camera", which holds here as in Swift because one is the other.
- **contract/**, **npy/**: byte for byte. The npy writer also re-encodes
  numpy's own four fixtures to numpy's bytes.

No tolerance was loosened.

## 2026-11-15 · Small differences from Swift, by construction (Phase 2)

- `Contour.levels` runs its levels in turn, not with `concurrentPerform`;
  each level is contoured whole, so the answer is the same.
- `Solid::checking` walks its directed edges in order (a `BTreeMap`), so
  which open edge a refusal names is fixed. Swift walks a dictionary there,
  whose order is per process, so its message can name a different edge
  from run to run. Only the message differs.
- `JsonError` now has Swift's cases (`NotAnObject`, `MissingKey`,
  `WrongType`, `BadSyntax`); a parse error is `BadSyntax`. Decode errors from
  any part of the schema are a `SchemaError` whose message is Swift's.
- Swift's `min(max(x, lo), hi)` is `clamp` here, which propagates a NaN as
  Swift's does. Elsewhere Rust's `f64::max` and `min` drop a NaN where
  Swift's `max` and `min` may keep it, by argument order (`AABB`, a grid's
  range). Only a NaN input tells them apart.
- `if !(x > 0.0)` stands for Swift's `guard x > 0 else`, so a NaN is
  refused as Swift refuses it; clippy's `neg_cmp_op_on_partial_ord` is
  allowed in `kurven-core` for that reason.

## 2026-11-15 · Where the Swift looks wrong (ported as it is)

- `DepthFrame.tile` builds its sub-frame with the default raster order,
  `.buffer`, whatever the frame's own order. A screen-order frame's tiles
  come back in buffer order. The bake tiles only buffer-order frames, so
  nothing draws wrong today.
- `Camera.perspectiveClip(fovY:aspect:near:)` takes `near` and never reads
  it (its doc says depth travels in the colour attachment); the port drops
  the parameter and says so.
- `Camera.pageCoordinates(of:)` under perspective divides by
  `max(v.z, .leastNormalMagnitude)`, but view z grows toward the viewer and
  a visible point has z < 0 (`perspectiveClip`'s w is −z), so every visible
  point is divided by 2.2e-308 rather than by its distance. Reproduction:
  the recip orbit with distance 10 and a 50° field of view; the point one
  unit right of the target in view space, at view z = −10, comes out at page
  x = 1 · cot(25°) / 2.2e-308 ≈ 9.6e307 instead of cot(25°) / 10 ≈ 0.214
  (`camera::tests::perspective_page_coordinates_divide_by_the_least_normal_as_swift_does`
  pins it). Nothing reaches it: the only caller in `Sources/` is a test
  (`TopEdge.swift`) on a bundle preset, which is orthographic; a bake
  refuses a perspective camera (`BakeError.perspective`), the app disables
  baking under it, a composite refuses it, and the preview projects on the
  GPU with `perspectiveClip`. Ported as it is.

## 2026-11-15 · Swift's pretty JSON is json.dumps's layout, not its bytes

`JSONValue.pretty(indent:ascii:)` says it writes what `json.dumps(...,
sort_keys=True, indent=..., ensure_ascii=...)` writes, byte for byte. Its
layout is that, but its strings and numbers are the canonical writer's:
`\b` and `\f` come out as `\u0008` and `\u000c` (Python: `\b`, `\f`), DEL
is left as it is under `ascii` (Python: `\u007f`), and a double from 2^53 up
to 1e16 takes an exponent. The port writes what Swift writes, and its doc
says so; no fixture index holds any of these.

## 2026-11-15 · Swift lets a malformed optional number fall back

`optionalDouble` answers nil for a key that holds anything but a number,
so the decoders that read with it take a wrong type as absent and use the
default: a polyhedron hatching's `step` ("fine" reads as the spacing), a
wall selector's `base` and `topOffset`, a real-band truncation's `beyond`,
an occluder's `base`, a projection's `yScale`. A refusal would name the
key; Swift's decoder does not refuse, so neither does the port, and a test
pins it (`a_misspelled_optional_number_falls_back_as_in_swift`). A
landscape's or a surface's hatching refuses any `step` before reading it.

## 2026-11-15 · The npy writer differs from Swift's past a 65,535-byte header

`NPY.write` always writes version 1.0 and puts the header's length down
as `n & 0xFF, (n >> 8) & 0xFF`, unchecked: a header past 65,535 bytes gets
a wrong length, and the file cannot be read. It takes a shape of some
twenty thousand axes, so nothing reaches it. The port does not write a
wrong length: past 65,535 it writes version 2.0, a four-byte length, as
numpy's `np.save` does, padded by numpy's rule (64 less the preamble and
header modulo 64). Below that the bytes are Swift's, and numpy's.

## 2026-11-15 · The wasm module grew to 503,030 bytes

From 86,969 in Phase 0; its code section is 455 KB. The golden generator
now links kurven-core: the manifest decoders with their refusal messages,
float formatting, contouring. The `web` profile is opt-level 3 with LTO; an
opt-level for size, and what the page's module exports, are Phase 6
decisions, so this is recorded, not tuned.

## 2026-11-15 · Phase 1: kurven-math against the oracles and against Swift

All of `Sources/KurvenMath` is in `crates/kurven-math`, a module per Swift
file. The gate is green at the fixtures' own tolerances, none changed:

- `tests/fixtures/expr/index.json`: all 66 cases. The worst against each
  case's tolerance: besselj(10, z) 3.7e-11 (tol 1e-9), bessely(2.7, z)
  2.3e-11, airyai and airybi 1.4e-12, expi 2.8e-13 (tol 1e-10), zeta 2.2e-14
  (tol 1e-10), tan_wide 2.2e-14 (tol 1e-12); everything else 1.1e-14 or
  less. Darwin's libm against the libm crate (the entry above: 1 to 3 ulp on
  one call in ten) makes no difference at these tolerances; there was no
  lane to stop.
- `language.json`: 41 canonical spellings (each a fixed point, with its
  reading's values on the grid), 17 refusals at the frozen character and
  length, the 43 functions with their arities, aliases and help, the 4
  constants.

As a second reference beyond the oracle grids, Swift's own KurvenMath was
compiled from `Sources/KurvenMath/*.swift` with `swiftc` and a probe main in
a scratch directory (nothing under `Sources/` changed), and evaluated at
the 4,845 points of the new `math/functions.txt` golden. The NaNs and
infinities are in the same places; the finite values agree to 5.1e-14
relative for Airy and 2.4e-15 for everything else.

## 2026-11-15 · Swift's near-miss message is not the one language.json froze

For `gama(z)` the fixture froze "did you mean gamma or rgamma or lgamma?"
(Python's `difflib.get_close_matches`, which breaks a tie in nearness by
taking the later name first). `Language.unknown` in Swift breaks it by the
name ascending and writes "gamma or lgamma or rgamma?", as `kurven-cli
surface periodic --expression 'gama(z)'` shows. The Swift suite checks only
the place of a refusal and that the message contains "gamma", so it never
saw this. The port writes Swift's message. Its test holds every frozen
message, which the Swift suite does not, and names this one as Swift's.

## 2026-11-15 · Where Swift traps, the Bessel functions answer NaN

`Bessel.swift` converts doubles to Int with `Int(x)`, which traps on a value
that is not finite or does not fit. The probe build exits on SIGTRAP (status
133) for `besselj`, `bessely`, `besseli` and `besselk` of order `1/0`, for
`besselj(1e300, z)`, and for `besselj` and `bessely` of order 2.7 or -2.3 at
z = inf (the Hankel path with an order of one or more). A landscape or the
app evaluating such an expression would crash. Rust's `as` saturates
instead, and the loops the result counts would then not end, so at exactly
those conversions the port answers NaN (`swift_int` in `bessel.rs`). This is
the one place the port's values differ from Swift's by construction; a unit
test holds it.

## 2026-11-15 · Phase 1's other differences from Swift

- **Characters.** Positions and lengths count `char`s (Unicode scalar
  values); Swift counts `Character`s (grapheme clusters). Name characters are
  `char::is_alphabetic` and `is_numeric`, where Swift asks `isLetter` and
  `isNumber`. On every source the fixtures hold, and every one the language
  accepts, the two agree. They can differ on text with combining marks, such
  as `e` followed by U+0301: Swift reads one name `é` and refuses it as an
  unknown name; the port refuses U+0301 as a character the language does not
  use, one character further on. Both refuse it.
- **The Fourier torus's sums.** `FourierTorus.fit` in Swift splits the
  samples over `activeProcessorCount` chunks, so the order of its sums, and
  their last bits, depend on the machine. The port sums the samples in order
  in one pass, as the handoff asks of every `concurrentPerform`, so its fit
  is the same everywhere. The residual is a maximum and does not depend on
  the order. The suite's bounds (1e-8 on the fit, 1e-7 on its derivatives,
  a ratio over 8 between the two lengths) are met.
- **SIMD.** Swift's `SIMD2/3/4<Double>` are `Simd<N>` (`simd.rs`), lane by
  lane with the scalar's rounding. Their `max()` reduction is IEEE maxNum
  (`Double.maximum`), which passes over a NaN lane; the ODE's error norm
  reads it, and the port keeps it.
- **ODE.** `ODE` is `Ode` (Rust's naming). Swift's `integrate` takes a
  `body` that may throw; no caller's does, and here it cannot, so the result
  is `Result<Statistics, Failure>`.
- **Names.** snake_case throughout. Where Swift overloads on argument type,
  the port has two names: `Trig::sin_pi` and `Trig::sin_pi_complex`. Default
  arguments are explicit (`Root::brent` takes `ftol` and `max_iterations`;
  `Root::brent_default` and `LambertW::w0` take Swift's defaults).
  `callAsFunction` is `Compiled::call`. Tuples are small structs
  (`Cylinder`, `Modified`, `AiryPair`, `Elliptic`, `Jet`).
- **The function table** holds plain function pointers (`Body::Unary`,
  `Body::Binary`); the tape calls them directly, and `Function::apply`
  serves a hand-built call of another arity, as Swift's `apply` does.
- **Constants computed at load** (log π, √π, e⁻¹, ζ's weights, Airy's ω) go
  through `kurven_math::real` once, behind `LazyLock`, where Swift computes
  them with Darwin's libm.
- **Complex** is not `Hash` (Rust's f64 is not), and its `Display` writes
  Swift's `description`, `(1.0-0.5j)`.

## 2026-11-15 · kurven-math's golden sets

Three files under `crates/fixtures/rust/math/`, written by `kurven-cli
fixtures` and, byte for byte the same, by the wasm module under Node:

- `functions.txt` (455,515 bytes): 57 expressions (every function; each of
  two arguments at orders 0, 2.7, -2.3 or moduli 0.2, 0.9; three of the
  operators and constants) at 85 points, as bits: a grid over the
  landscapes' windows, both sides of the branch cuts, the poles and branch
  points, far out on each axis, an infinity and a NaN.
- `language.txt` (1,886 bytes): canonical spellings, refusals and literals.
- `dynamics.txt` (4,329 bytes): the ODE integrator, lines and generator, a
  Fourier torus fit with its second-order jet and residual, Brent,
  continued fractions, quarter periods and `ellpj`.

With them the wasm module is 272,871 bytes (profile `web`).

## 2026-11-15 · Review of PR #64: corrections to the Phase 1 entries

- **Bessel is not the one place Swift traps.** ζ traps too: on a NaN
  imaginary part (the series order is `Int` of a NaN; Swift built from
  `Sources/KurvenMath` exits on SIGTRAP at (nan, nan) and (0.3, nan)), and
  in Euler-Maclaurin past |Im s| = 2^63. The port answers NaN there too. The
  policy lives in `kurven_math::swift`: `swift::int` is `Int(x)` as an
  Option. Every other cast from f64 to an integer in the crate is guarded as
  Swift's is (`Complex::pow` below |100|, the continued fraction below
  1e15, a literal's integer spelling below 1e15).
- **min and max.** Swift's generic `min(x, y)` is `y < x ? y : x` and
  `max(x, y)` is `y >= x ? y : x`, so a NaN in the first place survives
  (measured: `max(nan, 16)` is nan, `max(16, nan)` is 16); Rust's `f64::min`
  and `max` drop it. `swift::min` and `swift::max` are used wherever Swift
  calls min or max on doubles: the ODE's step control (its `hmax` was
  `f64::min`), Brent's step test, the frequency search and generator, the
  continued fraction's stop and ζ's order. Values differ from before only
  for NaN arguments.
- **Characters.** A CR LF pair is one `Character` in Swift, and was two
  positions here, so every place after a Windows line break was one late
  (Swift refuses `z\r\n+ @` at character 4). The tokenizer and the literals
  now count the pair once and keep it as written in a rewrite. What still
  differs is any other cluster of several scalars, such as a letter with a
  combining mark; the language refuses those either way.
- **Infinities in the oracle gate** are now held part by part with their
  sign, and NaN to NaN, where the Swift suite (and the port, ported from
  it) took any two values that are not finite as agreeing. The fixtures
  hold 9 such points (log and atanh at their branch points, ζ's pole, expi
  at 0, bessely at 0); the port has every one.
- **Sorting.** Swift's sort is stable (since Swift 5; a probe confirms it),
  not the unstable introsort the frequency module's comment said. Rust's
  is stable too, so ties order alike.

## 2026-11-15 · Review of PR #64: faster, with the same bits

Each change was checked to leave every bit as it was: the goldens
byte-identical, every value of the 66 oracle grids identical (a hash of
their bits, before and after), and J, Y, I, K and Airy identical at 19,602
points for orders up to 200, integral and not.

- Airy's Bessel form took I and K of order 1/3 three times over; it takes
  them once and forms I_{-1/3} by `modified`'s own formula.
- `Bessel::j` no longer computes Y beside J (Swift's does, every time); a
  negative order, which needs Y, still goes through `cylinder`. Y of an
  integral order takes J from its caller, its factorials from a table of
  the first 171 (the product is infinite after), and carries the harmonic
  numbers from term to term, the same sums in Swift's order. It was
  quadratic in the order, in Swift as here.
- The tape runs in a per-thread scratch buffer (`kurven_math::scratch`),
  neither allocated nor cleared per point; so do the Fourier torus's jets.
- Functions are found by hash, as Swift's dictionary finds them, and the
  tape's rare `Apply` holds its function, resolved when compiled.

The wasm module is 282,216 bytes after these.

## 2026-11-15 · A huge Bessel order or argument still hangs, in Swift too

Some Bessel loops run as many times as the order, or as |z|: Y of an
integral order n (the series' first sum, n terms), I and K (the
recurrence, about ν steps), and J and Y on the Hankel path with an order of
one or more (Miller's recurrence, about |z| steps). Measured here at
z = 1 + 0.5i: bessely(1e6) 3.7 ms, bessely(1e8) 0.21 s, besseli(1e8)
0.46 s, besselj(1e7, 2e7 + i) 0.10 s; so an order or an argument near 1e12
does not finish. Swift has every one of these loops, and Y's was quadratic
there: bessely(3000) takes 6.5 ms in Swift and 17 µs here, and Swift's
besselj goes through Y, so besselj(1e6, z) does not finish in Swift at all
(here it is 27 µs, the series alone). So no guard was added: it would give
an answer where Swift gives none, a choice for the owner. Until there is
one, a typed `besseli(1e12, z)` can hold the page's Worker, which Phase 6
should be able to cancel.

## 2026-11-15 · A NaN's sign bit differs by platform; the port reads it as plus

CI on x86_64 Linux wrote two lines of `math/functions.txt` otherwise than
the Mac: asin(inf + 0i) and acos(inf + 0i) had an imaginary part of -inf
and +inf on the Mac and NaN on Linux. The cause: asin forms 1 − z² part by
part, and its imaginary part −2 · inf · 0 is a NaN the hardware makes.
IEEE 754 leaves that NaN's sign bit to the platform: ARM's default NaN is
positive and x86's negative, and wasm passes the host's through, so the
wasm module agreed with native on each platform and the platforms
disagreed. `Complex::sqrt` then copied that sign onto an infinity
(`copysign(t, z.im)`), and on x86 the -inf it made became inf − inf = NaN
when the root was added to iz.

The rule: a NaN's sign bit is platform noise and decides nothing. Swift
runs only on ARM, where the NaNs an operation makes are positive, so the
port reads every NaN's sign as plus. `kurven_math::real` holds the only
reads of a sign bit: `copysign(x, y)` gives |x| when y is a NaN;
`sign_minus(x)` is Swift's `x.sign == .minus`, false for every NaN (so
-0.0 and -inf are minus); `total_order` sorts every NaN as the one
positive NaN. `clippy.toml` and `scripts/check-web.sh` refuse
`is_sign_negative`, `is_sign_positive`, `total_cmp` and `copysign` on f64
and f32 everywhere else.

Why that is enough: nothing else in the engine reads a NaN's sign bit.
Comparisons are false for a NaN of either sign; `abs` clears the bit;
`as` casts a NaN to 0; Rust prints a NaN without a sign, `description`
writes `nan`, and the goldens write `nan`; `to_bits` is used only to print
a number that is not NaN, and in tests guarded by `is_nan`. Of the libm
crate's functions only `copysign` returns a number decided by a NaN
argument's sign; the others return a NaN for a NaN argument, or a value
(pow(1, NaN), hypot(inf, NaN)) that does not depend on its sign. x86 could
not be run here; the unit tests pass NaNs of both signs through the three
functions and through `Complex::sqrt`, and hold asin and acos at inf + 0i
to the Mac's values.

The one way this can differ from Swift: a NaN that Swift negates (`-x`
flips the sign bit) or that descends from such a one keeps its minus sign
on ARM, and Swift's `.sign` reads it as minus; the port reads it as plus.

## 2026-11-15 · The NaN-sign rule covers the merged kurven-core

After merging Phase 2, kurven-core's six sign reads (-0.0 in the JSON
writer and in Solid's text, three sorts by `total_cmp`, a test) go through
`real::sign_minus` and `real::total_order` as well. None could see a NaN,
so nothing changed; the goldens are byte-identical. kurven-core's other
bit read, `ulp` in `surface_request.rs`, takes `abs` and refuses a value
that is not finite first, so it reads no NaN's sign either.

## 2026-11-15 · Phase 3: the landscape's core, what came across

Ported into `kurven-core`, a module per Swift file: Surface.swift
(`surface.rs`, beside the `ContourRefine` already there), ScalarField.swift
(`scalar_field.rs`: `ScalarField`, Swift's `Contour.levels(of:...)` and
`Contour.clipped` as the free functions `levels` and `clipped`, the plain
routine and its chord refinement, trimming, `TrimIndex`, `short_way`),
Face.swift (`face.rs`, `Solid::face` included, built from what `solid.rs`
has), FaceContour.swift (`face_contour.rs`), Mesh.swift (`mesh.rs`),
Parametric.swift (`parametric.rs`, every type whole), Bundle.swift
(`bundle.rs`) and Pick.swift (`pick.rs`), which no wave named: a face's
boundary cuts and chains its pieces with `Face.pieces`, `snapped` and
`chained`, and a parametric face reads its foot and seams from it. Grid.swift
was already whole. Hatching.swift has nothing a landscape reaches that is
not in `hatching.rs`: the rest lays hatching on a solid's faces, Phase 5's.

Of Scene.swift, `scene.rs` has the types (`ContentID`, `Heightfield`,
`SceneGeometry`, `Scene`, `PreviewMode`, `Lighting`) and these methods:
`Heightfield.init`, `forked`, `onGrid`, `forEachSample`, `normal`,
`graphNormal`, `surfaceHeight`, `drawnHeight`; `Scene`'s four initializers,
`heightfield`, `parametric`, `solid`, `looking`, `drawing`, `creases`. The
fold curves, the visibility of curves and layers, `GridTraversal`,
`FoldPlacer`, the bounds and `parallelMap` are the next wave's and are not
declared.

## 2026-11-15 · Hatch.swift stands in as hatch_stub.rs until its wave

Surface, Mesh, Face, FaceContour and Bundle call into Hatch.swift, a later
wave. `hatch_stub.rs` holds what they call, every item marked `PORT: phase 3
stub (Hatch.swift)`, and the file goes whole when `hatch.rs` lands. The small
pure helpers are ported as Swift has them, so their callers are right
meanwhile: `cell`, `edgeTolerance`, `point(on:at:)`, `point(from:to:at:)`,
`clip`, `replicate`, `HatchContext` and `tileBoundaryCrossings`. The rest
answers empty: `topEdge` (so a wall face's boundary is its bottom and ends
only), `tileBoundaryStretches` (no riser mesh at a tile boundary),
`plateauContours`, `riserStretches`, and `curves`' hatchings, wall edges,
truncation edge and integral curves. `graphTrimLoops` answers the domain's
edge, which is all of an untruncated graph's loops; a truncated graph lacks
its plateaus' loops until the port, so its faces' contours are not cut at
the truncation edge yet. `curves` dispatches as Swift's does otherwise: a
contour of |f| or arg f to `derive`, of a coordinate or the height to the
face contours.

## 2026-11-15 · Phase 3 core: differences from Swift, by construction

- **Shared grids.** `Surface` holds its grids behind `Arc`. Swift's arrays
  are shared until written, so a face's closures capture the surface by
  value for nothing; a clone of the port's surface shares its samples the
  same way. `restyled` shares them with the bundle it came from.
- **Names.** `Face.Copy` is `FaceCopy` (`Copy` is a Rust trait); `Computed`
  makes its value once under a `OnceLock`. Swift's defaulted arguments are
  structs whose `Default` is Swift's defaults: `Trim` for the trim loops,
  `contains`, cuts, excess and lift of `Contour.levels`, `FaceInit` for the
  tiled face initializer (`Face::single` is the one-copy one). Overloads
  take two names: `Surface::height_at`/`height_at_tiles`,
  `lift`/`lift_path`, `KurvenBundle::layer`/`layer_at_levels`,
  `levels`/`levels_resampled`, `SceneGeometry::face_of` (Swift's
  `face(of:)`), `face_at`, `boundary_of`; `Scene`'s initializers are
  `heightfield_scene`, `parametric_scene`, `solid_scene`, `from_bundle`.
- **Scene's identity.** `ContentID` counts from one per process where Swift
  mints a UUID; a memo needs only uniqueness. `Scene`'s geometry, layers and
  identities are private with getters, since a public Rust field could be
  reassigned and leave `content` stale, the premise Swift's `let` keeps.
- **Preconditions.** `ParametricSurface::new` and `with_positions` answer
  `ParametricError::NoVolume` (or a position count) where Swift's
  precondition traps; `ParamAxis::new` asserts two samples, as
  `Grid2D::new` asserts its count, every caller computing a floor of 2.
- **Where Swift's `Int(x)` traps.** On a point that is not a number:
  `drawnSurface` and `clampedHeight` answer NaN, `interpolate` a point of
  NaNs, `TrimIndex` files the point in its first bucket, `Face.pieces` in
  its first cell, `snapped` in a cell of its own; a tile boundary riser or a
  vertical face whose sample count does not fit takes 2, a picked boundary's
  segment 4096 steps. No finite input reaches any of these.
- **Concurrency.** `parallelMap` and `concurrentPerform` are loops in
  order; each piece is independent, so the answers are the same.
- **Mesh.** Triangles are `[i32; 3]`; `concat` offsets them with wrapping
  addition, as Swift's `&+` does. `Lighting` holds its direction as
  `[f32; 3]`, normalized as `simd_normalize` does.
- **Writing a bundle.** The manifest is written to a temporary file and
  renamed into place, Swift's `atomically: true`.

## 2026-11-15 · Phase 3 core: the suites, and how two of them stand in

Ported, with Swift's tolerances: Fields.swift all but its torus case
(seven passing; the torus case ignored as Phase 4's, expressed against
`ParametricSurface::torus`); Chords.swift's "the plain routine's chords
stay within its tolerance" (its isophote part split off and ignored as
Phase 5's) and "cuts on a domain whose edge the loops follow"; Faces.swift's
"each layer says the kind of face its curves lie on" and "a polyhedron's
faces are its loops, and membership agrees with the convex pieces" ("a
parametric surface is one face" ignored as Phase 4's); `main.swift`'s
restyle suite, its wall hatch ignored until Hatch.swift. The `core:` suites
were already ported in Phase 2. New: the landscape half of "contract:
bundles decode into consistent values", the four landscapes of
`tests/fixtures/expr` reading and building their scenes and faces, and a
described bundle written and read back the same.

Two stand-ins. Swift builds the chord suite's landscapes with
`NativeLandscape.build(..., refine: true)`, which needs the refiner; the
faces it contours read f only through `ContourRefine.magnitude`, so the
port's landscape is the grid of |f| with |f| in closed form as that
magnitude and contours that move nothing. Swift names its polyhedra
(`Solid.platonic`, `Solid.frame`, `.csaszar`, Phase 5's); the port reads the
same seven solids from the contract's polyhedron fixtures, and the face-kind
suite carries sn's layer specs on a plane patch, since the rule reads only a
layer's role and source.

## 2026-11-15 · The restriction suite, over the grid's landscapes

Faces.swift's "restriction: a contour's copy is cut to the footprint where
it lands, exactly" is ported over the landscapes of Swift's
`faceLandscapes()` drawn on the grid (`kurven-core/tests/restriction.rs`):
the contract's bundles and the hatch fixture's twenty variants on
footprints, reflected, turned and in four tiles. It passes at Swift's
tolerances (1e-9 of the domain for a kept point or an end off the
perimeter). The landscapes on f (Γ, the towers, the reviewer's tower) are
built with `NativeLandscape.build(..., refine: true)` and wait for the
refiner.

## 2026-11-15 · Phase 3: the refiner, the catalog, the sampling, the memo, the depth clip

Ported whole: `Refine.swift` to `kurven-landscape/src/refine.rs`, but for
`NativeLandscape.refiner(for: bundle)` and `benchmarkRefinement`, which
read a `KurvenBundle` and its `Surface` (the core wave); `Catalog.swift` to
`catalog.rs`; `Landscape.swift` to `landscape.rs`, but for
`NativeLandscape.build`, which builds the bundle; `ContourMemo.swift` to
`kurven-core/src/contour_memo.rs`; and the depth-buffer half of
`HiddenLine.swift` (`foldCrossing`, `HiddenLine.clip` and `pass`,
`DepthImage.outline`) to `kurven-core/src/hidden_line.rs`.
`HeightfieldVisibility` waits for the grid traversal. `RefinementReport`
and `summarize` are ported, so the benchmark is a loop over them.

One stub: `short_way` in `refine.rs`, Swift's free `shortWay` from
`ScalarField.swift`, which the core wave ports; it goes at the merge.

Spelled otherwise: Swift's refiner keeps f as four closures (`value`,
`magnitude`, `phase`, `turnedPhase`) and passes one as `g`; here they are
methods over one tape, and `g` is a `Scalar` (magnitude, phase, turned
phase). The `refine` property is `refine_hook()` (the method `refine(field:
...)` keeps the name); `zero(near:)` and `pole(near:)` are `zero_near` and
`pole_near`; `turn(_:around:)` and `turn(between:_:)` are `turn_around` and
`turn_between`; `ended(atSingularPoints:)` is `ended`. Default arguments are
explicit. `concurrentPerform` over paths, rows and edges is a loop: each
iteration writes its own slot.

## 2026-11-15 · The handoff discs in a fixed order, where Swift walks a Set

`ContourRefiner.refine` collects the singular points whose rays reach the
handoff radius in a `Set` and maps it to the discs, and a point inside two
discs (two singular points closer than six cells) is cut back to the first
disc that holds it: in Swift that is the `Set`'s per-process order, so the
drawing can differ from run to run there. The port walks the points in
ascending order. Nothing else in the refiner depends on a hash order:
`unaccounted` sorts its cells, `drawnOnce` only asks whether any segment is
near, and the memo forgets by a clock.

## 2026-11-15 · Where the refiner traps or never returns in Swift

- **A field that is not f's.** `field(_:)` is `preconditionFailure` for u,
  v, the height and an isophote. Here `ContourRefiner::field` answers None;
  `refine` hands the paths back as the grid placed them, and
  `position_errors` answers NaN for every vertex. No caller asks.
- **A NaN ceiling.** `refine` copies itself with the call's ceiling and span
  and recurses whenever they differ from its own, and NaN never equals NaN,
  so a phase line asked under a NaN ceiling recursed until the stack ran
  out. The port always places on a copy, which gives the same curves
  wherever Swift returns, and returns for NaN too.
- **`Int` of a coordinate.** `drawnOnce` bins vertices by `Int` of x / cell,
  `snap` takes `Int` of the grid position, `seededRings` of the zero's.
  Each traps on a value that is not finite. A vertex that is not finite is
  binned in a cell of its own, `snap` leaves such a vertex where it is (as
  it does one off the grid), and `seeded_rings` passes such a zero over.
- **The landscape's rounding.** `pyRound`, and so `gridShape` and
  `wallDensity`, trap on a domain whose side is not a number. The port's
  `py_round`, `grid_shape`, `wall_density` and `sample` answer None.

## 2026-11-15 · A zero exactly on a node is not a singular point (ported as is)

`root(near:of:)` refuses a start where g is zero (`guard g0.magnitude > 0`),
and a zero that falls on a grid node is found only from that node, the
least of |f| beside it. So z³ on [-1, 1]² at 41 samples has no singular
point: its phase lines are carried onto the zero by `ended` rather than
seeded from the normal form. Ported as is. The port's analytic tests of
rays and the branch cut shift the window by a quarter cell so the zero
lies inside a cell.

## 2026-11-15 · The memo's keys, and a NaN in them

Swift's `ContourMemo.Key` compares the paths as `Hashable` points do, by
`==` (so -0 is 0), and the level, ceiling and span by their bits. The port
does the same, hashing each coordinate plus 0.0 so -0 and 0 hash alike. A
path with a NaN coordinate makes a key unequal to itself, which Swift's
dictionary stores and never finds again; Rust's `HashMap` needs a key equal
to itself, so the port does not store such a key and solves it each time,
which is the answer Swift gives. `Synchronization.Mutex` is
`std::sync::Mutex`, a poisoned lock read through (an entry is inserted
whole). Forgetting by least recent use sorts by a clock no two entries
share, so which entry goes is fixed in both.

## 2026-11-15 · Phase 3 refiner tests and their tolerances

Swift's, with Swift's tolerances: "refine: a zero or pole of order more
than one is found" and "... the grid's nodes do not see is found, by the
winding number of its cell" (1e-9 cells), "... by an accumulation of zeros
or poles" (sin(1/z) and 1/sin(1/z) at 400), "refine: a phase line is placed
only where it can be drawn" (0.05 cells; tan at 240). Each built the
bundle and asked `refiner(for: bundle)`, which reads the canonical
expression, the domain and the height grid's shape; the port asks
`refiner(expression, domain, grid_shape(...))`, the same refiner, and takes
the phase grid from `sample` and `quantize` as `build` does. Also "landscape:
the derived numbers", "... are the committed ones" (hatch/index.json),
"expression: the language table is the frozen one" through the catalog, the
three "clip:" suites (bit for bit), and the half of "landscape: built here,
it is the committed bundle" that needs no bundle: the four committed
landscapes' grids at Swift's 1e-6 relative, their manifests' shape,
truncation, walls, occluder step, projection and layers exactly. All pass,
none ignored.

The port's own, against closed forms: |z| = r circles, every vertex on the
circle and every chord's sagitta within the stated 0.02 cells; |z³ − 1| = c,
three ovals and one curve, chords within 0.02 cells by the refiner's own
estimate; a ring round a zero the grid cannot see, on its circle; the phase
lines of z³ as three rays from the zero a third of a turn apart, on their
rays to 1e-9; the branch cut of z and of z³ drawn once, as one and three
lines; Newton, Schröder and the winding number on z²(z − 0.5)/(z + 0.3i)³;
the hook's contours the refiner's, remembered and forked to the bit.

One thing the closed forms showed: a vertex is not always on the level to
the root finder's tolerance. Where the float32 grid puts a vertex within
rounding of a node, `snap` leaves it there if its residual over the gradient
is under a hundredth of the tolerance (`1e-2 * tolerance * cell`): on |z| =
0.8 at 41 samples one vertex is 1.2e-8 off, 2.4e-7 cells. That is Swift's
rule, so the circle test holds vertices to that bound, and nine in ten to
1e-9.

## 2026-11-15 · Phase 3: slope lines and the landscape builder

Ported whole: IntegralCurves.swift to `kurven-core/src/integral_curves.rs`
(`VectorField` with its constant field's closed form, the integrator and
its stopping rules, `Face::slope`, `height_slope`, the slope raised by the
graph's metric, `Seeds::points` and `closes`, `Surface::integral_curves`
and `graph_integral_curves`); `NativeLandscape.build` to
`kurven-landscape/src/landscape.rs`; Refine.swift's `refiner(for: bundle)`
and `benchmarkRefinement` to `refine.rs`. That empties Landscape.swift and
Refine.swift of PORT holes. Lines of curvature and every integral curve on
a parametric surface are LinesOfCurvature.swift's, Phase 4's; on the graph
the principal directions answer no field, as in Swift.

Spelled otherwise: Swift's `IntegralCurve` namespace is the module, so
`IntegralCurve.End` is `integral_curves::End`, with `Section`, `Limits`
and `Traced` beside it; `Face.raised(_:overGraphOf:)` and `raised(_:_:)`
are `raised_over_graph` and `raised`; `refiner(for: bundle)` is
`refiner_for_bundle`, beside the expression's `refiner`. Default arguments
are explicit (`Limits::default()` is Swift's defaults). `build` answers a
`BuildError`: Swift's thrown compile error, or a domain whose side is not
a number, on which Swift's rounding traps.

Merge note: `Surface::curves`' stub for `LayerSource::IntegralCurves`
(`hatch_stub.rs`) becomes `PolylineSet::from_paths(&self.integral_curves(field,
seeds, faces, &context, refine, &Limits::default()))`, as Hatch.swift's
`curves` calls it. `integral_curves.rs` takes `HatchContext` from
`crate::hatch_stub`; the import moves to `crate::hatch` when Hatch.swift
lands.

## 2026-11-15 · Slope lines: differences from Swift, by construction

- **Parallel seeds.** Swift traces a copy's seeds with
  `Heightfield.parallelMap`, each thread making a worker of its own: a
  forked refiner (`ContourRefine.fork`) and the graph and field on it. The
  port makes one worker per traced copy and traces its seeds in turn. A
  fork is f compiled again and its hook is pure, so the curves are the
  same.
- **Where Swift traps.** `Seeds.points` by arc length counts its seeds with
  `Int(ratio.rounded())` and then `0..<n`, which traps for a negative n: a
  spacing below zero, which `Seeds.unsupported` refuses when a manifest is
  decoded but a hand-built `Seeds` can hold. The port places no seed there.
  The `cut!` of the first return cannot be nil (it is set at the first
  return, before any later one), and the port reads it without a panic.
- **The benchmark is native alone.** `benchmark_refinement` times with
  `std::time::Instant`, which panics on wasm32-unknown-unknown, so it is
  compiled out there. Nothing on the page asks for it.

## 2026-11-15 · Slope lines: the suites, and what the golden draws

IntegralCurves.swift's suites, at Swift's tolerances, none loosened (in
`kurven-landscape/tests/integral_curves.rs` where they build a landscape
with the refiner, `kurven-core/tests/integral_curves.rs` where they do
not): slope lines of |f| on their phase contours on sn, the rational
function and exp(z) (0.02 cells); each stopping rule (a zero 1e-12, the
truncation edge 1e-12, a pole within 1e-3 cells, the domain's edge and a
footprint 1e-12, closed loops of arg f's slope 0.02 cells, the step limit
at eleven vertices, a seed off the domain); the slope of a coordinate
raised by the metric (1e-8, the adjugate 1e-12) and exp(20z) followed to
the edge; seeds once each and no closer than a cell; seeds by arc length
(1e-9) and on tiled graphs (1e-9, 1e-12, sn 1e-5); the manifest; the
constant field's flow to the bit. All pass. Two are ignored: "on f, seen
exactly where the dense cast against f says" needs Scene's visibility (a
parallel wave), and the torus half of the linear-curve suite needs
`SurfaceDrawing.build` (Phase 4).

The slope-lines golden (`tests/fixtures/fields/landscape_slope_lines`),
read and given `refiner_for_bundle`'s hook as the bake gives it, traces 24
curves, each from a zero of (z² − 1)/(z² + 1) to the truncation edge or the
domain's edge, on its phase contour to 0.02 cells, but two: arg f = −π runs
along the real axis from each zero into f's saddle at 0 and ends there,
singular, 6.5e-6 from it (8e-5 cells), where the field vanishes. That is
the integrator's rule, not a defect; the test holds those two to the true
contour alone, as Swift's sn suite holds the lines through sn's critical
points, since the drawn contour at ±π leaves the crossing open. The drawn
comparison waits for the bake.

Expressions.swift's bundle suites now run: "landscape: built here, it is
the committed bundle" whole (the manifest byte for byte, the samples to
1e-6, every layer's path count and length to 1e-6; both sides derive their
hatchings and edges through `hatch_stub` until Hatch.swift lands), and
"refine: contours placed by the function", "... phase contours end on the
zeros of f", "... the branch cut is drawn once", "... a restyle places only
the curves it has not placed before", "... a chord on the level set, split
for its sag, is placed" (`kurven-landscape/tests/refined_bundles.rs`). All
pass. With `build` in, the chord suite's and the restriction suite's
stand-ins (the landscapes on f) can be built as Swift builds them.

## 2026-11-15 · Phase 3: the landscape's visibility, what came across

The rest of Scene.swift a landscape reaches is in `kurven-core/src/scene.rs`:
`Heightfield.foldCurves`, `endedAtZeros`, `visibleFolds`, `stop`,
`surfaceDepthTolerance`, `onSurface`, `visibleCurves` (every face kind),
`visibleWallCurves`, `facing`, `wallFacing`, `riserFacing`, `Tests`,
`visibleRuns`, `turning`, `parallelMap` (both forms), `FoldPlacer` whole,
and `Scene.visibleLayers`, `visible`, `viewBounds`, `quickBounds`.
`GridTraversal` is `traversal.rs`, public, since the bake asks it.
Occlusion.swift is `occlusion.rs`: the `OcclusionTest` trait, the
traversal's conformance, and `SphereTracing` whole, ported ahead of Phase 4
because it reads only a `TubularNeighbourhood`, whose type is here; nothing
builds a surface with one until the torus, so it is untested until then.
`HeightfieldVisibility` joins the rest of HiddenLine.swift in
`hidden_line.rs`. Visibility.swift is `visibility.rs` and SurfaceImage.swift
`surface_image.rs`: their types whole, their behaviour Phase 4's.

The signatures the bake reads: `Scene::visible_layers(&self, Option<f64>,
Option<&SurfaceImage>) -> Result<Vec<(Layer, PolylineSet<ViewSpace>)>,
PortRefusal>`, `Scene::view_bounds(&self) -> Result<Option<AABB<ViewSpace>>,
PortRefusal>`, `Heightfield::fold_curves(&self, &Transform<WorldSpace,
ViewSpace>, refined: bool) -> PolylineSet<WorldSpace>`, `GridTraversal::new(
Heightfield, &Transform<WorldSpace, ViewSpace>, depth_tolerance: f64)`. A
landscape's layers never refuse but for a hatching on the graph.

## 2026-11-15 · The landscape's visibility: holes, each a typed refusal

- `Scene::visible` on a parametric surface, phase 4: coordinate hatching,
  its fold curves, sphere tracing or the coordinate image, the image's
  depth for curves with no coordinates. `view_bounds` refuses a parametric
  surface with fold curves among its layers (it places them).
- A hatching on the graph, phase 5: `Surface.graphHatching` is in
  Isophotes.swift, which the plan puts in Phase 5, though it lays a
  landscape's hatching. No committed landscape or gallery bake has one.
- A solid's clipped edges, its hatching and a clipped file's curves on it,
  phase 5 (`Solid.visibleEdges`, `visibleHatching`, `visibleCurves(onSurface:
  occlusion:)`); its unclipped layers are drawn whole, as in Swift.
- `Solid::occlusion`, phase 5. `Solid.Occlusion`'s type was not in
  `solid.rs`; `SolidOcclusion` declares its fields, and its `OcclusionTest`
  conformance is left out (not an empty impl), so nothing can ask it.
- `SurfaceVisibility`'s footprint, front-most test, sheet, `is_visible`,
  crease and fold crossing, `HiddenLine::clip_on` and
  `SurfaceImage::rasterize`, phase 4. Its facing and `is_clear` are whole.

## 2026-11-15 · The landscape's visibility: differences from Swift, by construction

- **Workers.** `parallelMap(_:chunk:worker:_:)` makes its worker once and
  maps in order (`parallel_map_with`); Swift makes one a thread. A forked
  refiner answers as the one it came from, so the drawing is the same.
- **Tests.** Swift's `Tests` holds three closures, here boxed, each worker's
  sharing its forked traversal behind an `Rc`.
- **Refusals.** `visible_layers`, `visible` and `view_bounds` answer a
  `Result`, since a scene can reach a deferred branch; Swift's preconditions
  in `SurfaceImage.init` and `SurfaceVisibility.init(sphereTracing:)` are
  error values (`CoordinateCount`, `NoNeighbourhood`).
- **Where Swift's `Int` traps.** The traversal's crossings of a grid family
  take `Int` of a bound that is NaN for a point that is not a number, and the
  fold placer's grid edges `Int` of its index: the family, or the edge, is
  passed over. No finite input reaches either.
- **The trace.** `explain` writes Swift's lines with C's `%.4g` and `%.3e`
  done by hand (Rust's formatting has neither); it is a diagnostic, and no
  test compares its text.
- **Defaults.** Swift's defaulted arguments are explicit: `fold_curves`
  takes `refined` (true in Swift), `visible_folds` and `visible_curves`
  take `resolution` (0), `quick_bounds` its budget (20,000), `unoccluded`
  its flags (`unoccluded_default` is Swift's `unoccluded(_:)`).

## 2026-11-15 · A lone tile is read untransformed (ported as is)

`GridTraversal.drawnSurface(at:)` and `Surface.tile(under:tiles:)` treat a
heightfield of one tile as the domain itself: the point is read where it
stands, not carried back through the tile. A heightfield whose one tile is
a translation or a reflection would be tested against the domain under the
world point, not under its image. Every manifest's one tile is the identity
(a native landscape's, and each fixture's), and the traversal's grids do
carry the lone tile's inverse, so only the height read differs. Ported as
is.

## 2026-11-15 · The landscape's visibility: the suites, and what stands in

Stress.swift, all six suites, at Swift's tolerances: the 4x4 ray cast
(over 40,000 decidable points, none disagreeing), the facing converging
(under 1% at 128 cells), the reflected and turned tiles on f against the
walk, the real drawing against the brute-force cast (14,384 facing vertices
on the grid, 14,213 on f, none disagreeing), scaling by 8 and by 1/8
(worst 0 exactly) and the bake at every resolution. Two stand-ins. The bake
suite compares bakes at four depth resolutions; nothing of the visibility
reads a depth buffer, so it holds `Scene::visible_layers` the same every
time it is asked, the drawing drawing (230 paths, 541 units on rgamma,
without the hatching and edges the stub leaves out) and the steep case's
kept length. The real-drawing suite also counts what a 1000-pixel depth
buffer hid and asks that the pit at -4 hides more by the traversal than by
it; there is no rasterizer in the core, so that half is left out and the
traversal and the cast are held equal in the pit instead.

Occlusion.swift: the landscape part (10,000 and more points, none
disagreeing); the polyhedra (Phase 5) and the torus (Phase 4) ignored.
Facing.swift: the ridge's facing and folds, the pits (its bake check reads
the fold layer of `visible_layers`), and the navigation check. Faces.swift:
the tile-boundary riser, the three fold suites, and the folds' half of "a
contour or a fold carried to the truncation edge across a band boundary
ends on the drawn top". 12 tests pass, 2 are ignored.

The oracles spread over threads with `std::thread::scope` in the tests
alone. Single-threaded, the port draws rgamma at 240 refined in about a
third of a second.

Waiting for Hatch.swift's strokes (each tests hatching, wall edges, the
truncation edge or the top edge): "walls: the hatching and edges of a cut
face are tested by the wall", "walls: a stroke on a wall is tested by the
wall it stands on, on a notch and at a tile boundary", "walls: a stroke is
hidden up to where the walk up the line of sight meets terrain, on abutting
copies and across a notch", "risers: a riser's hatching and bottom edge lie
on it, from its bottom to its top, where it stands", "risers: only a layer
that selects the risers hatches them, in place or on every tile", "risers:
a riser's hatching and bottom edge are hidden up to where the walk up the
line of sight meets the solid", "risers: no stroke is drawn through a
riser, and none is hidden that the walk up the line of sight sees",
"plateaus: the hatching on a plateau is tested by the plateau", "walls: a
vertical stroke is its two ends, however high |f| stands", and the contour
layers' half of "faces: a contour or a fold carried to the truncation edge
across a band boundary ends on the drawn top".

## 2026-11-15 · Hatch.swift is hatch.rs; the stub is gone

`Sources/KurvenCore/Hatch.swift` is ported whole into
`kurven-core/src/hatch.rs`, and `hatch_stub.rs` is deleted. The stub's
items keep their signatures (`HatchContext::new`, `GraphTrimLoops`,
`RiserStretch`, `TileBoundaryStop`, `Surface::cell`, `edge_tolerance`,
`point_on`, `point_from_to`, `clip`, `replicate`, `tile_boundary_crossings`,
`graph_trim_loops`, `top_edge`, `tile_boundary_stretches`,
`plateau_contours`, `riser_stretches`, `curves`); what was empty now draws.
Added beside them, Swift's other items under their snake_case names:
`hatch`, `stroke_count`, `vertical`, `subdivided_profile`, `with_minima`,
`top_edge_height`, `wall_hatch_lines`, `wall_edges`, `plateau_hatching`,
`runs_at_or_above`, `bisected`, `truncation_excess`, `truncation_edge`,
`drawn_truncation_edge`, `band_riser_stretches`,
`tile_boundary_riser_stretches`, `riser_hatch_lines`,
`banded_truncation_edge`, `on_boundary_truncation_crossing`, `Rectangle`
and `PlateauLoops`, and `HatchContext::with_perimeter` for Swift's
defaulted initializer. Hatch.swift's `truncationCrossing(from:_:to:_:excess:)`
is `truncation_crossing_between`, since `truncation_crossing` is
Surface.swift's overload on a lifted segment.

Two holes. `curves` keeps integral curves as the stub the IntegralCurves
wave replaces at the merge. `graph_hatching` (a `.hatching` on the graph)
is Swift's `Surface.graphHatching`, which lives in Isophotes.swift beside
`Face.coordinateHatching`, `Face.graded` and `Solid.graded`, and reads a
half-Lambert tone off each copy's outward normal: Phase 5's, refused with
`PORT: phase 5`, drawing nothing. No landscape fixture or reference
drawing has a graph hatching.

## 2026-11-15 · Hatch.rs: differences from Swift, by construction

- **Closures that recurse or share state.** Swift's nested functions
  become small structs: `Station` (the top edge's tuple), `Emitter`
  (`bandedTruncationEdge`'s `emit` with the curves, the pieces by band and
  the ends on each boundary), `PartSubdivider` (its `subdivided`), and free
  functions for the two `subdivide`s. `PlateauLoops` borrows its plateau
  test for a lifetime instead of holding a `@Sendable` closure.
- **Dictionaries keyed by points or doubles** (the plateau loops' ends, the
  stretches' meeting points, the ends on a boundary) key by the bits with
  -0 read as 0, as Swift's `==` and `Hashable` have them; a point with a
  NaN is not a key, as Swift's dictionary never finds one. The stretches by
  band are a `BTreeMap`, Swift's `keys.sorted()`.
- **Sorting doubles** goes through `real::total_order` of `x + 0`, so -0 and
  0 tie and keep their order as Swift's stable sort by `<` does; a NaN sorts
  last where Swift's order is unspecified.
- **Where Swift's `Int` traps.** `strokeCount` on a count that is not a
  number or does not fit answers 2 (its answer for a spacing that is not
  one); `onBoundaryTruncationCrossing` answers None for a vertex whose grid
  index is not a number. No finite input reaches either.
- **`Set([0, count - 1])`** in the uniform truncation edge is iterated as
  `[0, last]`; each end is solved alone, so the order does not matter.

Ported as Swift has them: `bandedTruncationEdge` takes `beyond` and never
reads it (`_beyond`); the rings' `drawn` segments are gathered once per
band before that band's rings are cut, while `endsDrawn` reads the pieces
as they grow; a tile boundary's riser edges seek no minimum of |f| between
stops, as their doc says.

## 2026-11-15 · Hatch.rs tests, and how the landscapes on f are built

Ported with Swift's tolerances, all passing. In `kurven-core/tests`:
`hatch.rs`, main.swift's four `hatch:` suites (every committed stroke of
`tests/fixtures/hatch` to 1e-9, the truncation edge segment for segment to
1e-9, the plateaus' contours bit for bit, the whole gaps the walls meet)
and Hatchings.swift's "one rule for a manifest's step and angles";
`truncation.rs`, Truncation.swift's three suites ("the grid is the
rasterizer's" needs no GPU: it is `Mesh::truncation_edge_mesh`);
`restyle.rs`'s wall hatch, no longer ignored. In `kurven-landscape/tests`:
`trim_loops.rs`, Faces.swift's eleven suites from "close a plateau that
turns back a column short of a band boundary" to "a pole beside a band
boundary, at every resolution" (492 poles), and "the trim loops and the
truncation edge the same in whatever order the tiles are listed";
`faces.rs`, "the graph's trim loops are its truncation edge, closed, and a
contour on it is cut where it crosses them" over 63 landscapes;
`tiling.rs`, "a landscape's faces tile its boundary", "walls: a wall on a
band boundary reads its footprint's band" whole, and the two `depth:`
suites; `chords.rs`, Chords.swift's truncated-graph suite for u, v and the
height, its isophote and graph hatching families ignored as Phase 5's.

Stand-ins. Swift builds every landscape on f with
`NativeLandscape.build(_:refine: true)`, which is the IntegralCurves wave's;
`common::build` makes what the suites read from it the way it does (the
float32 grids from `sample` and `quantize`, `ContourRefiner::new(...)
.refine_hook()`, and for Γ its walls on `rectangle(domain, wall_density)`).
The contract bundles' faces are built from their occluder directly,
which is what `Scene(bundle:preset:)`'s geometry does. "a bundle's occluder
holds the risers" needs a bundle with the build's occluder: its manifest is
a contract landscape's with the build's domain, truncation and occluder,
all that `KurvenBundle::occluder` reads. No suite was skipped for needing
Scene's fold curves or visibility.

## 2026-11-15 · The suites' landscapes on f are built by NativeLandscape::build

The stand-ins the earlier waves recorded are gone, and no result changed.
`common::build` (the visibility and stress suites) is now
`NativeLandscape::build(request, "native", refine)`. `support::build` (the
face suites) calls it too, with `layers: []`. The surface and the refiner do
not depend on the layers, and deriving the default layers through the
refiner would cost a landscape on f most of its build for curves those
suites never read. The plain routine's chord suite no longer uses a
closed-form |f| with a do-nothing hook. It builds "z" and "z^2" over [-2, 2]²
as Swift does (`layers: []`, the occluder's context) and has moved to
`kurven-landscape/tests/chords.rs`, beside the other chord suite.
`kurven-core/tests/chords.rs` keeps the cuts suite, which builds no
landscape. Every suite that used a stand-in passes as before.

## 2026-11-15 · TopEdge.swift and Facing.swift's strokes: every suite passes

`kurven-landscape/tests/top_edge.rs` holds TopEdge.swift's fourteen suites
and `walls.rs` holds Facing.swift's nine on the strokes Hatch.swift draws.
All use Swift's tolerances and all pass. No suite turned up a porting bug.
None of them bakes. Facing.swift's "walls: the hatching and edges of a cut
face are tested by the wall" ends by baking at 400 and comparing the wall
hatch stroke for stroke. Here it compares that layer of
`Scene::visible_layers`, passed through `HiddenLine::pass`, which is a
landscape's bake stroke, against `visible_wall_curves`. The walks up the
line of sight (the walls', the risers' and `SolidWalk`) are brute force, as
in Swift, and spread over threads per camera with `common::par_map`, each
camera's tally independent. `walls.rs` runs in about 12 s. TopEdge's banded
suite has a loop over the riser's chords that adds `min(...) * 0.0`, which
does nothing. It is ported as a comment, and the sag is measured below as in
Swift. `densely_hidden` and `per_unit` move from `stress.rs` to
`common/mod.rs`, so the slope-line suite can read them.

## 2026-11-15 · PolylineSet::total_length is planar; Swift's suites measure in space

`PolylineSet::total_length` ignores z, because it is the length a drawing
comparison compares. The TopEdge and Facing suites sum segment lengths in
space (`simd_length` of 3D differences). The pit of 1/Γ traces 0.294 units of
fold in the plane and more than 0.5 in space, so the first port, written
with `total_length`, failed on the test and not on the engine. The suites
now use a 3D length of their own. Later ports should check which length a
Swift suite means before using `total_length`.

## 2026-11-15 · The last Phase 3 suites, and what stays unported

All of these pass at Swift's tolerances:

- IntegralCurves.swift's "on f, seen exactly where the dense cast against f
  says", with Swift's `SegmentIndex` ported into the test.
- The contour layers' half of Faces.swift's "a contour or a fold carried to
  the truncation edge across a band boundary ends on the drawn top", which
  joins its folds' half in `visibility.rs`. It checks 2,000 to 32,000
  vertices a layer on the towers and on Γ at 600.
- Faces.swift's restriction suite over the landscapes on f
  (`kurven-landscape/tests/faces.rs`). These are the towers in one tile and
  in two, and the reviewer's tower. Γ and the pole a tenth outside the tile
  boundary stand on the whole region, which Swift's own filter passes over,
  so the earlier entry's "Γ" was never in this suite.
- Hatchings.swift's three suites on the value types: Turing's pattern, a
  tone or step refused at decode, and the round trip through fields
  (`kurven-core/tests/hatch.rs`).

Not ported, each for a reason:

- Faces.swift's "a polyhedron face's trim loops are its loops, and a contour
  cut at them is its hatching" reads `Solid.hatchLines`, which is Phase 5.
  It is ignored in `kurven-core/tests/faces.rs`. Hatchings.swift's six
  solid suites are Phase 5 as well. So are Chords.swift's isophote parts:
  "on the graph of sin z" is isophotes throughout, and is ignored whole.
- In main.swift, "bake: a perspective camera is refused" belongs with the
  bake crate, and "preview: curves on a surface steep in view ..." needs the
  GPU preview. "bake: a tiled bake draws what a single-pass bake draws" has
  nothing to hold here. A landscape's strokes are `visible_layers`, which
  reads no depth buffer, so tiling the buffer cannot change them. The stress
  suite already holds `visible_layers` the same every time it is asked.
- Depth.swift's "the Metal pass agrees with an exact ray cast" is the GPU
  depth pass, Phase 7.

## 2026-11-15 · Phase 3: the bake and the CLI's landscape commands

Bake.swift is `kurven-bake/src/lib.rs`: `BakeOptions` (with `tile_count`),
`BakeError` with Swift's two messages and a third case for what the port has
not brought across, `Bake` and `bake`. No stroke of a landscape reads the
depth buffer (`Scene::visible_layers` decides), so the port renders none: a
`Bake` holds the strokes and the specs, and Swift's `depth`, `surface` and
`tiles` are left out. The outline (traced from the buffer), a parametric
scene (its coordinate image comes from the depth pass) and `bake --dump`
(it writes the buffer's frame) are typed refusals, `PORT: phase 4`, until
the CPU rasterizer arrives with `SurfaceImage`. The perspective and empty
scene refusals are Swift's, checked in Swift's order. kurven-bake depends
on kurven-core alone; kurven-surfaces is Phase 4's and joins when the
parametric bake does. Swift's suites "a perspective camera is refused" and
the strokes' half of "a tiled bake draws what a single-pass bake draws"
are ported (`kurven-bake/tests/bake.rs`); the depth half waits for the
rasterizer.

Swift's perspective message is a multi-line literal whose source lines
were joined by hand, so it holds four runs of thirteen spaces ("indexes by
an             affine map"). The port writes it as it is, and a test pins
it.

`kurven-cli` gains `landscape`, `bake` and `references`, read with Swift's
`Args` (`args.rs`) and refusing as Swift refuses (exit 1, `kurven-cli:
<error>`). `landscape` is whole (presets, interval, bands, isophotes and
hatching on the graph, slope lines, `--repeat`); its console summary names
the layers and does not describe their sources (Swift's `sourceName` and its
`%g`), and prints the truncation as Rust's Debug: stdout is not compared.
`bake` and `loadDrawing` read landscape bundles; a surface's bundle (`PORT:
phase 4`) and a polyhedron's (`PORT: phase 5`) are told apart by the
manifest's top-level key as Swift tells them and refused with a message.
Numbers on the command line are read by Rust's `f64` parser where Swift
calls `Double(String)`, which also reads C's hexadecimal floats; no flag is
given one.

`references` bakes what is ported of Swift's list: the six gallery
landscapes (make.py's manifest edit, one sampled cubic for the four corners,
cn's quarter periods by the AGM, its words written with Swift's
`description` so each `how` is byte for byte Swift's), the four native
landscapes and the slope lines. sn, sn in relief, the torus, `periodic` and
the frame's two gallery drawings (Phase 4), the seven solids and the
composite (Phase 5) are left out of the list with a PORT comment. It takes
`--only NAME,...`, which Swift's does not, so a single drawing can be
measured.

## 2026-11-15 · The landscape references against Swift's: all but one layer at rounding

*Superseded 2026-11-16: ang_major is at rounding too, in all eleven drawings; see "The six phase-line drawings were the port's Float.pi".*

`kurven-cli references` bakes eleven of Swift's reference drawings (the six
gallery landscapes, the four native landscapes, the slope lines), and
check-web holds each to Swift's under its line of
`crates/fixtures/references.txt`. Every drawing has Swift's layers in Swift's
order. In every layer but one, the stroke counts are Swift's and the Hausdorff
distance is 0 at the SVG's four decimals, but for one coordinate printed a
digit apart in the four cubic corners' wall_hatch (1.000e-4) and the slope
lines (1.117e-4, length 1.05e-7 relative). So every drawing is held to
1.5e-4 page units, the floor two drawings equal to rounding can meet
(1e-4 on each axis, 1.414e-4 on the diagonal), and to 1e-5 of each layer's
length: a fifteenth of the provisional 1e-3 that compare defaults to.
compare reads both drawings from their SVGs, which hold page coordinates
in the plane, so its lengths are planar on both sides, as a drawing's are;
it does not use `PolylineSet::total_length`.

## 2026-11-15 · The phase lines Swift and the port draw differently

*Superseded 2026-11-16: the two causes below are wrong; the cause was the half turn `contour::cut` turned the grid by; see "The six phase-line drawings were the port's Float.pi".*

The one layer that differs is ang_major (arg f at 0, ±π/2 and ±π) in six
drawings, all at the ±π level, for two causes, each found by building
Swift's KurvenLandscape in a scratch package with the refiner's stages
printed and comparing them with the port's stage by stage (the grid's
contours and the singular points agree; the difference starts inside the
refiner).

- **A saddle of f.** Where f' = 0 the level lines of arg f cross: 1/(z³ − 1)
  at z = 0, where the real axis and the rays at ±60° meet, and cn(z, 0.64)
  at 2K and 2iK'. The chord across the crossing is placed or not by
  rounding (the residual over a vanishing gradient), and a piece that ends
  there is followed on f along whichever branch it meets. In the cubic Swift
  splits the grid's path at z = 0 (pieces ending at (−0.00055, 0.00095) and
  starting at (0.00037, 0.00064)) and follows each half on over ground
  another curve covers, so it draws the −60° ray and the ray to the pole at
  e^{−2πi/3} twice, 6.1 world units of ink twice over; the port places the
  chord and draws each stretch once. In cn the pieces through 2iK' pair up
  the other way.
- **The ±π cut along the window's edge.** Γ and cn are real on the real axis,
  which is the window's lower edge in landscape-gamma and gallery-cn, so the
  cut lies on the edge, where Im f is rounding noise, and the poles (Γ) and
  zeros (cn) on it are found a hair to either side: Newton puts Γ's pole at
  −2 at Im z = +1.3e-38 in Swift and −1.7e-40 here, inside the window in one
  and outside in the other, so its rays along the edge are drawn in one and
  not the other. In cn Swift keeps the stretch of the edge from K to 2K in
  the up-going phase line and the port cuts it there, so Swift draws it
  twice.

Both are a decision rounding makes at a degenerate point, as a bisection
falls one way or the other at a tangency, not a run the port fails to
place: every stretch Swift draws once, the port draws once (the cubic's six
rays, cn's edge from K to 3K), and the difference is a stretch drawn twice,
a pairing at a crossing, or rays of a point on the edge. Measured (Swift
against the port): gallery-cn 19 strokes each, Hausdorff 2.18e-4, length
3.19e-2; the corners tm90s0.5 13 against 12, 1.55e-3, 6.87e-2; tm90sm0.5 12
against 11, 1.11e-4, 4.98e-2; t90s0.5 11 against 10, 1.17e-4, 1.03e-1;
t90sm0.5 11 against 10, 1.37e-3, 1.44e-1; landscape-gamma 8 against 7,
1.13, 2.68e-1. The corners' 1.4e-3 is Swift's doubled stroke crossing the
saddle by a chord, which the port's strokes turn at. So that layer is
skipped in those six drawings (`compare --skip 1`, measured and reported but
left out of the verdict); every other layer of each, and every layer of the
other five, is held. It would take an analytic reference to say which side
of the edge a pole exactly on it belongs to; Swift's drawing does not
decide that, so nothing in the port was changed.

## 2026-11-15 · Bake times, and the landscapes' goldens

Measured on this machine (Apple Silicon, 18 cores), from the request to
the strokes, each plate built, edited as make.py edits it and baked in
memory (`kurven_golden::gallery::bake_plate`, the same strokes as the
references): native release, single-threaded, gallery-gamma 12.4 s and
gallery-cn 46.3 s; the wasm module (profile `web`) under Node 26, 36.2 s and
65.0 s (`crates/kurven-wasm/bench.mjs`, through the export
`bake_gallery_plate`). Swift's `landscape` then `bake` of the same two
(without the hatching edit) takes 0.9 s and 3.5 s of wall clock on all
cores, 11.2 s and 54.4 s of CPU: the port does the same work in one thread,
as the rules ask, and a Worker per core is Phase 6's question. Through
`references --only`: a cubic corner 0.6 to 0.8 s, the four native
landscapes 0.02 to 0.15 s, the slope lines 0.06 s. The references step adds
about a minute to check-web, nearly all of it cn and gamma.

`kurven-golden` gains `landscape/`: the cubic at 64 and Γ by bands at 90,
built on their function and baked in memory (so the wasm module bakes them
too), each as its SVG and as each layer's stroke and vertex counts, length
bits and an FNV-1a hash of its vertex bits, which hold native and wasm to
the bit where the SVG holds them to four decimals. The two take 0.5 s
natively. With the bake and the gallery linked in, the wasm module is
1,567,692 bytes (from 704,201 after Hatch.swift merged).

## 2026-11-16 · Phase 3 status: what is ported, what is left, every hole

*Superseded 2026-11-16, its gate paragraph: ang_major is no longer skipped in any drawing; see "The six phase-line drawings were the port's Float.pi".*

**Ported** (Swift file to Rust module). kurven-core: Grid, Mesh (`mesh.rs`),
Surface with ContourRefine (`surface.rs`), ScalarField (`scalar_field.rs`),
Face (`face.rs`), FaceContour (`face_contour.rs`), Hatching's drawing half
for landscapes (`hatching.rs`), Hatch (`hatch.rs`), Parametric's types
(`parametric.rs`), Scene whole for a landscape (`scene.rs`, with
`GridTraversal` in `traversal.rs`), Occlusion (`occlusion.rs`, with
SphereTracing whole but untested), Visibility's types (`visibility.rs`),
HiddenLine whole (`hidden_line.rs`), SurfaceImage's type
(`surface_image.rs`), ContourMemo (`contour_memo.rs`), Bundle
(`bundle.rs`), IntegralCurves (`integral_curves.rs`), Pick (`pick.rs`).
kurven-landscape: Refine (`refine.rs`), Catalog (`catalog.rs`), Landscape
with `build` (`landscape.rs`). kurven-bake: Bake, strokes only. kurven-cli:
`landscape`, `bake` (landscape bundles), `references` (the eleven landscape
drawings), beside `fixtures` and `compare`.

**The gate.** 271 tests pass and 9 are ignored, each for Phase 4, 5 or 7 (the
suites agent's table, "The last Phase 3 suites, and what stays unported",
names the Rust test of every Swift suite). The eleven landscape reference
drawings are compared with Swift's by `crates/fixtures/references.txt`:
every layer at 1.5e-4 page units and 1e-5 of length, but ang_major in six
drawings, skipped with the reason in "The phase lines Swift and the port
draw differently". In five of them Swift draws more ink over the same
ground (a stretch twice); in landscape-gamma it is the port that draws
more, 4.13 page units of rays along the window's lower edge (Im z = 0,
where Γ's pole at −2 lands either side by 1e-38), 8 strokes in Swift
against 7 here.

**Left for Phase 4** (each a typed refusal marked `// PORT: phase 4`): the
parametric bake and the depth buffer's outline (`kurven-bake/src/lib.rs`),
`SurfaceBundle.read` and `bake --dump` (`kurven-cli/src/commands.rs`), the
surface references (`kurven-cli/src/references.rs`), a parametric surface's
layers in `Scene::visible` (`scene.rs`), the rasterizer
(`surface_image.rs`), the parametric fold curves, grid normals,
interpolated normals, linear curves, the two tori and the displacement
(`parametric.rs`), and SurfaceVisibility's footprint, front-most test,
sheet, `is_visible`, `onto_crease`, `fold_crossing` and `clip_on`
(`visibility.rs`). Torus.swift, Cover.swift, LinesOfCurvature.swift and
KurvenDynamics are not started.

**Left for Phase 5** (`// PORT: phase 5`): hatching a solid's faces
(`hatching.rs`, five places), Solid.Occlusion (`occlusion.rs`), a solid's
layers in `Scene::visible` (`scene.rs`), `PolyhedronBundle.read`
(`kurven-cli/src/commands.rs`), the solid and composite references
(`kurven-cli/src/references.rs`), isophotes (`face_contour.rs`), and
hatching on the graph (`hatch.rs` and `scene.rs`). The last is a landscape
feature: Swift keeps `Surface.graphHatching` in Isophotes.swift with
`Face.coordinateHatching`, so it waits for that file. No landscape
reference drawing uses it, and a bundle that asks for it is refused when
it is drawn.

No `PORT: phase 3` stub is left in the code; `hatch_stub.rs` is gone.

## 2026-11-16 · The CLI refuses a value option given bare, where Swift runs on its default

Swift's `Args` cannot tell a switch from an option whose value was left
off: `--preset` at the end of a line, or before another option, lands in
`switches`, which no reader of a value looks at, and `only` accepts it, so
`bake B --preset` bakes the first preset and `landscape z --truncation -o
x.kurven` samples untruncated, saying nothing. A command line that ignores
what was typed is not behaviour worth porting, so the port differs on
purpose: `Args::SWITCHES` lists the eight switches Swift reads with
`switchedOn` (conformal, draft, editing, no-refine, no-truncation, preview,
soft, visibility), and `only` refuses any other option given bare, "--preset
wants a value", as `switched_on` refuses a switch given a value. `bake
--outline` and `--dump` are refused as Phase 4 in either form, before
`only`: bare, they used to parse as switches and bake on, exit 0.

## 2026-11-16 · A bundle draws its described layers when first asked

Swift's `KurvenBundle` derives every described layer when it is made:
`read` derives them on the grids, then `loadScene` attaches the refiner and
`refined` derives them all again, the first set thrown away; `landscape`
built a bundle with its refiner, derived every layer through it, and wrote
the manifest and two grids. The port's `KurvenBundle::layers()` derives on
first call, once (a `OnceLock`), from the bundle's own surface, occluder
and refiner; `layers` is no longer a public field. What each layer is does
not change: a dumped layer is the one read, `refined` and `restyled` carry
a dumped layer by name as Swift's `redrawn` finds it (by spec, when a
restyle keeps it), a restyle keeps a described layer already drawn whose
spec and inputs did not move, and every other is drawn as Swift draws it.
The goldens and the eleven references are unchanged, and a test counts
the refiner's calls: none until the layers are asked for, then once.

## 2026-11-16 · `landscape` builds no refiner, and refuses its flags before it samples

Swift's `landscape` builds the bundle with its refiner and writes it, and
`KurvenBundle.write` writes the manifest and two grids, in which the
refiner changes nothing. The port builds without one (and draws no layer,
above), so the bundle written is byte for byte the same, which five
command lines checked against the previous build (the gamma preset, cn at
600, gamma banded, the cubic with slope lines, isophotes and hatching, and
exp(1/z) with a seed spacing); gamma's preset took 0.13 s and now 0.01 s.
`--repeat` builds the same way, so it no longer times a refiner. Swift also
reads `--isophotes`, `--hatching` and the seed spacing after the build, so
a mistyped flag cost a whole build to report; the port reads them first.
Only the hatching's default spacing reads the bundle (the plateaus'), so
the shading layers are made again with it once built; the seed spacing is
held to the shape `grid_shape` gives the build. Where a flag and the
expression are both wrong, the port names the flag and Swift the
expression.

## 2026-11-16 · The six phase-line drawings were the port's Float.pi

The ang_major difference in gallery-cn, the four cubic corners and
landscape-gamma ("The phase lines Swift and the port draw differently")
was a constant, not rounding. Swift's `Contour.cut(of:)` turns the float
phase grid by `$0 > 0 ? $0 - .pi : $0 + .pi`, where `.pi` is `Float.pi`,
which Swift rounds toward zero: 0x40490FDA, 3.1415925. The port's
`contour::cut` used `f32::consts::PI`, rounded to nearest: 0x40490FDB,
3.1415927. With Swift's (`contour::SWIFT_FLOAT_PI`), ang_major of all six
is at 0 page units and 0 of length with Swift's stroke counts, the skip is
gone from `references.txt`, and every layer of the eleven drawings is held.
The port's goldens moved with it: core/contour.txt's cut, and ang_major's
last length bits and vertex hash in landscape/cubic.txt and gamma-bands.txt.
The port has no other `f32` π; the Swift outside KurvenMetal that a grep
shows using `.pi` among floats is this one, though a grep cannot see every
place Swift infers `Float` for `.pi`.

**How it was found.** Swift's KurvenLandscape copied into a scratch package,
and the port's refine.rs into a scratch crate, both printing every input
path, snapped vertex, chord estimate, split and trace step as bit patterns,
run on the same bundle files. The first difference was the refiner's input:
the grid's paths of the cut, a vertex's coordinate apart by about 1e-7 of a
cell. Phase 3's entry looked inside the refiner and described where the two
routes parted, not why. With the constant fixed the two logs agree stage by
stage to the last bits of f.

**Why Swift's constant is the right one, analytically.** `quantize` clamps
a phase grid into the samples' own range, so a sample of exactly ±π (f real
and negative) is stored as ±0x40490FDA, the largest float under π. Turned
by that same float, it is exactly 0: on the level of the cut, whichever the
sign of Im f's zero decided ±π. Turned by π rounded to nearest, it was
±2.4e-7 by that sign. That is what happens on a window's edge on the real
axis, where Γ and cn are real: the grid drew the cut along the edge, or
not, as the zero's sign fell, and the port's 4.13 page units of rays along
landscape-gamma's edge were that. Test: `kurven-landscape/tests/refine.rs`,
`the_cut_along_the_windows_edge_does_not_hang_on_the_sign_of_a_zero`
(z − 1 and −(1 − z), whose arg on the real axis is π and −π, draw the same
cut; it fails with the nearest π). `kurven-core`'s
`the_cut_turns_the_grid_by_swifts_float_pi` pins the constant.

**Corrections to the Phase 3 entry.**

- *The cubic's saddle.* Swift does not split the path at z = 0 by rounding.
  The grid's path from the 60° branch to the 0° one is subdivided across
  the saddle, and its last chord, between a vertex on the 120° branch and
  one on the 60°, is 2.5e-4 off the level at the depth limit, 0.026 of a
  cell against the 0.02 tolerance: no last bit decides that. The old port
  placed the chord because its cut's vertices started elsewhere.
- *Γ's pole on the edge.* Newton still puts the pole at −2 at Im +1.33e-38
  in Swift and −1.72e-40 in the port, and it changes nothing: `in_window`
  allows 1e-9 of a cell, so it is in the window in both, and under the
  truncation a pole seeds no rays.
- *cn.* The same constant; the pairing at 2iK' and the stretch K to 2K are
  now Swift's.

**What both now draw at the degenerate points, and Swift's defect.** The
port and Swift draw the same, and in two places it is not what f says:

- *A stretch drawn twice.* The cut of 1/(z³ − 1) through its saddle at 0
  has six branches; every one is drawn, but the two pieces of the path
  split at the saddle are each followed on f across it, onto the 300° and
  240° branches another run already draws. `drawnOnce` drops only a run
  lying wholly along another, so both stay: in the plane, 2.34 of the
  2.35 units of the run along those branches are drawn again (1.39 by one
  piece, 0.94 by the other). In gallery-cn the zero at 3K's ray runs along
  the lower edge through the saddle at 2K to the zero at K, and the grid's
  run from K along the edge to 2K and up to the pole at 2K + iK' covers K to
  2K again: 1.99 units twice. Both are in Swift's reference SVGs.
  Reproduction in Swift: build the corners' bundle (`kurven-cli landscape
  "1/(z^3 - 1)" --re -1.3,1.6 --im -1.2,1.5 --res 300 --truncation 5 -o
  c.kurven`), then `NativeLandscape.refiner(for:)`'s `refine(field: .phase,
  level: .pi, Contour.cut(of: phase), points: singularPoints(), ceiling:
  pow(2, 2.5))`, the rung the hook asks for 1.1 × 5: four runs of 2.345, 2.251, 3.123 and 1.882 units, the third and
  fourth lying along the first for 1.392 and 0.938. In the port the test
  `the_cut_through_a_saddle_draws_every_branch_and_two_of_them_twice_as_swift`
  counts the branches at 0°, 60°, ..., 300° drawn 1, 1, 1, 1, 2, 2 times.
  The change Swift would need, and the port with it in the same commit so
  the gate still holds them to each other: `drawnOnce` by the stretch, not
  the run (cut out of a run each stretch of two or more vertices lying
  within a tenth of a cell of a longer or earlier run, splitting it there),
  or a `traced` that stops where a followed end comes within a tenth of a
  cell of a run already kept. Either draws every branch once; the six
  references would be rebaked on both sides.
- *The cut along the window's edge.* Analytically the cut of the closed
  window includes the whole edge where f is real and negative. The grid
  draws it only on the side of each saddle of f where the window beside the
  edge is over the level (its tie: a node on the level is not over it), and
  a line followed on f along the edge (cn's from 3K) draws it whole. Swift's
  suite leaves rays along the edge optional ("refine: phase contours end on
  the zero" asks five of seven rays of each zero of 1/Γ, "the two along the
  real axis lie on the window's edge"), and the wall's top edge (wall_edges)
  draws that edge at |f| in every landscape, so the ink is there either
  way. It no longer hangs on a sign; a rule (say, a phase run along the
  window's edge is left to the wall's top edge) is the owner's to make, in
  both.

**The last bits left on this path.** The ceiling the hook is asked under
is the memo's rung, 2^2.5 for 1.1 × 5: Darwin's pow gives the correctly
rounded 0x4016A09E667F3BCD, the libm crate one ulp under. It only decides
whether a vertex's |f| is over the ceiling, and none of the six drawings has
one within an ulp of it.

## 2026-11-16 · The cut clamps its grid, the cut golden is quantized, the rungs are Darwin's

From the review of PR #67.

- **`contour::cut` clamps to Float.pi (a port-side addition).** A sample on
  the cut turns to exactly zero only when the grid was clamped as
  `quantize` clamps a landscape's, to ±`SWIFT_FLOAT_PI`; Swift's
  `Contour.cut(of:)` assumes that and says nothing. The port's `cut` now
  clamps every sample into [−Float.pi, Float.pi] before turning it (a NaN
  stays NaN), so π rounded up, from a grid not quantized, also lands on
  the level whichever its sign. On a quantized grid it moves nothing: the
  goldens are byte for byte and the eleven references at 0, as before. Swift
  would need the same clamp only for a grid that did not come through
  `quantize`; none does today.
- **The cut golden.** `core/contour.txt` stored its phase grid as `a as
  f32`, unclamped, so on the column x = 0.5, where the two angles sum to π
  either side by rounding, samples were ±0x40490FDB, and under Float.pi
  the cut zigzagged ±2.4e-7 along it (path 0 grew from 21 vertices to 28):
  the golden pinned the very sign-of-rounding behaviour Float.pi removes.
  It is now stored by `NativeLandscape::quantize`, as the engine stores a
  landscape's, and the cut runs straight up the column again (21 and 20
  vertices). The clamp above alone gives the same file.
- **The memo's rungs are Darwin's.** The ceiling the hook is asked under,
  2^(k/4), was `real::pow`, an ulp under Darwin's correctly rounded pow at
  2^2.5 (the "last bits left" of the entry above, now gone). It is read off
  2^(j/4), j = 0..3, correctly rounded, times 2^q, which is exact for a
  normal answer; under the normal range that would round twice (k = −4091
  and −4089 came out an ulp under), so there the subnormal is rounded once
  from 2^(j/4) to 62 bits. A scratch probe printed Swift's `pow(2, k / 4)`
  for every k a finite positive height reaches, −4296 to 4096, and twelve
  heights' rungs: all are the port's to the bit. A test pins eight of them.
  The goldens and references are unchanged; the ceiling decides only
  whether a vertex's |f| is over it.

## 2026-11-16 · Phase 5: the polyhedra and composites, what came across

Ported, a module per Swift file: the rest of Solid.swift into `solid.rs`
(the mesh, `transformed`, `resting`, `convex_hull`, `platonic` with the
`Platonic` enum, `csaszar`, `frame`, `visible_edges`, `visible_curves`,
`visible_curves_on_surface`, `along`); `Solid.Occlusion` into
`occlusion.rs` as `SolidOcclusion` with its `OcclusionTest` conformance;
Hatching.swift's solid half into `hatching.rs` (`hatch_lines`,
`hatch_lines_on_face`, `graded`, `hatch_lines_by_width`,
`visible_hatching`); PolyhedronBundle.swift into `polyhedron_bundle.rs`;
Isophotes.swift whole into `isophotes.rs` (`Face::is_flat`,
`outward_normal`, `isophote`, `isophotes`, `hatch_levels`,
`coordinate_hatching`, `graded`; `ParametricSurface::coordinate_hatching`
and `isophotes`; `Surface::graph_hatching`, moved out of `hatch.rs`, where
Phase 3 kept its stub); Composite.swift into `composite.rs`, Slab.swift into
`slab.rs`, Contact.swift into `contact.rs`. `Scene::visible` draws a
solid's clipped edges, its hatching by width step and a clipped file's
curves on it, and a graph's hatching by width step; `Face`'s isophote
contours on the graph (`face_contour.rs`) draw. kurven-cli gains `surface
solid` (`kurven-cli/src/solid.rs`, with `surface_command`, which sends
every other shape to Phase 4) and `scene` (`kurven-cli/src/scene.rs`,
`composite_drawing`); `bake` and `load_drawing` read a polyhedron's bundle.
No `PORT: phase 5` is left in the code.

Spelled otherwise: Swift's static `Slab.tops(_:_:origin:_:_:size:tolerance:)`
is `Slab::sides_on_tops`, beside `tops` of a cell; `contact(onFace:of:...)`
and `contact(on:...)` are `contact_on_face` and `contact_on_surface`;
`sweep(ofBall:...)` is `sweep_of_ball`; `simd_double2x2` is `slab::Mat2`
(its inverse the adjugate over the determinant); `zeros`'s nested
functions, which recurse, are the methods of a struct (`Zeros`).
`Composite.visibleLayers` answers a `CompositeError`, and a part whose own
layers the port refuses (a parametric surface, Phase 4) is refused through
it by the refusal's text. `Solid.HatchLine`'s width step is an i64.

## 2026-11-16 · Phase 5: differences from Swift, by construction

- **Where Swift's `Int(x)` traps** (all on values no finite input
  reaches): a face's range of hatch lines past Int lays none; `graded`'s
  count of readings past Int reads once; `hatch_levels` on a periodic axis
  past Int lays none; a slab's strip whose footprint is not finite passes
  the copy over; `Slab::cells` past Int reaches none; `chains` keys a point
  that is not finite as a node of its own; `circles`' chord count past Int
  is one chord.
- **Orders.** Every dictionary and set Swift iterates here is sorted there
  already (`hatchLinesByWidth`'s and `visibleHatching`'s keys, the
  isophote hatching's steps), so the port's `BTreeMap`s give Swift's order;
  the sets and dictionaries that are only asked (`diagonals`, `seen` in the
  hull and in `chains`, `chains`' buckets) are hashed. `platonic`'s
  `Array(Set(scaled))` then sorted is deduplicated in order, then sorted
  the same way.
- **Threads.** `Composite.cut`'s `parallelMap` over segments and
  `contact(on:)`'s over cells and pieces are loops in order, each item
  independent. The descent over blocks of cells, a recursion in Swift, is
  a stack visited in the recursion's order, so the cells, the pieces and
  the chained curve come out in Swift's order.
- **simd.** `simd_double3x3(simd_quatd(angle:axis:))` (`resting`) is the
  quaternion `(sin(θ/2) axis, cos(θ/2))` through simd's quaternion-to-matrix
  formula, term for term, in `solid.rs`.
- **A stub's type.** Phase 3's stub of `VisibleHatching` held its visible
  spans in view space; Swift's are in world space, and so are the port's.
- **Float π.** Nothing ported here uses `Float.pi` (Swift's is 0x40490FDA,
  not `f32::consts::PI`); every π is a double.
- **ParametricSurface's hatching and isophotes are here.** Isophotes.swift
  holds `ParametricSurface.coordinateHatching` and `isophotes`, so
  `isophotes.rs` ports them; Phase 4's `Scene::visible` on a surface should
  call these, not port them again.
- **`SurfaceRequest.unsupported` is not ported.** It lives in
  KurvenDynamics' SurfaceDrawing.swift (Phase 4). The isophote manifest
  suite checks its first clause through `SurfaceManifest::unsupported` and
  leaves its hatching-vertex rule to an ignored Phase 4 test.

## 2026-11-16 · Phase 5: the drawings against Swift's

Each at Hausdorff 0 and length 0 at the SVG's four decimals, with Swift's
stroke counts, so each is held to the 1.5e-4 / 1e-5 floor the landscapes
are (`references.txt`): the five Platonic solids, Császár's torus and the
frame (5, 9, 8, 20, 18, 15, 18 edges); gallery-frame-edges (18) and
gallery-frame-turing (18 edges, 796, 767, 635 and 314 hatch strokes by
width). The composites Swift's fixtures hold without a torus,
`tests/fixtures/composite/cube.svg` and `frame.svg`, drawn by the port's
`scene` path, are at 0 too, every layer and count (kurven-cli
`tests/composite.rs`, the same floor). composite-three is not baked: its
torus part is a `SurfaceBundle`, Phase 4's. `references::composite_three`
draws it and joins the list when that merges.

Bake times, single-threaded, in memory (`kurven_golden::polyhedra::
bake_named`; native by the ignored `bake_times` test, wasm by `bench.mjs`
under Node 26): the dodecahedron 0.001 s native, 0.002 s wasm; the frame
under Turing's pattern (2530 strokes) 0.61 s and 0.90 s; the composite of
the cubic at 48 on its grid with a cube and a frame (627 strokes) 0.028 s
and 0.042 s. Through the CLI, `scene` of the cube on the 120-sample cubic
fixture takes 0.053 s. The wasm module is 1,782,351 bytes.

## 2026-11-16 · Phase 5 status

**Ported**: everything the handoff's Phase 5 names, and every `PORT:
phase 5` hole (none is left). **Goldens**: `crates/fixtures/rust/polyhedra/`,
ten drawings with summaries, written alike by native and wasm.

**Suites at Swift's tolerances, all passing**: Solids.swift whole (15 tests),
Hatchings.swift's six on a solid, the polyhedra of Occlusion.swift,
Faces.swift's polyhedron hatching, Isophotes.swift but its torus parts (6
pass), Chords.swift's isophote parts (3), Composite.swift's three (the
composite gate takes about 90 s on all cores). No tolerance was loosened.

**Left for Phase 4** (`PORT: phase 4`): the contact line on a parametric
surface (`Composite::visible_contact`, Swift's `clipOnChords`, which needs
SurfaceVisibility's `isVisible` and `test`); composite-three; `surface`
for the parametric shapes (`kurven-cli/src/solid.rs`'s `surface_command`);
the torus halves of the isophote, composite and contact suites, each an
ignored test saying what it holds.

**Contact.swift** (`kurven-landscape/tests/contact.rs`): five suites pass
(the contact line on the terrain and its part over the eight gate
compositions, the box sunk into a plane, a part floating, the sweep of
polyhedra, a face flush with the terrain) and five torus parts are ignored
for Phase 4. Swift's "over 150 parts met" counts the tori with the
polyhedra, so it is held in the ignored torus sweep; the polyhedra alone
are held to their crossings (over 100,000, none missed). The grid half of
"isophotes and tone: seen exactly where the torus's own cast and the grid
landscape's cast say" passes; its torus half is ignored.

## 2026-11-16 · Where the Swift looks wrong (Phase 5, ported as it is)

Both found in review of PR #68; Swift has each, so the port keeps it, a
test pins it, and the fix is the owner's call.

- **The floor of a footprint that is not convex.** `Slab.floorSpan` finds
  the span of a segment whose lines of sight cross the slab's floor in each
  of the footprint's convex pieces, then returns their hull (`best.map {
  (min($0.0, lo), max($0.1, hi)) }`), not the pieces' spans. On a footprint
  with a notch the stretch between two pieces' spans, whose lines cross the
  floor's plane outside the footprint, is reported hidden, and
  `Composite.cut` erases another part's ink there. Reproduction: a flat
  terrain at 1 on the L of [-1, 1]² less the quarter x > 0, y > 0 (17
  samples, no walls, so base and top are 1), seen from below along +z; the
  segment at z = 0 from (-0.5, 0.8) to (0.8, -0.5) is under the L for
  s ≤ 0.385 and s ≥ 0.615 and under the notch between, but its floor span
  is [0, 1] and nothing of it is seen
  (`contact.rs`, `the_floor_of_a_footprint_that_is_not_convex_hides_the_notch_as_swift_does`).
  It takes a line of sight that reaches the floor from below, so another
  part under a footprint with a notch, seen from below the landscape.
  Returning every piece's span (the cut merges them) would fix it.
- **A change of width on a vertex.** When a line's width step changes
  exactly at a vertex b, `Solid.graded` bisects the change to b itself (the
  step at every double below b is the old one), and the interval it would
  start there is empty and dropped. `Face.graded` therefore keeps the old
  step, appends b, and meets the change again at the next segment's start
  (t = 0), appending b a second time: the piece before ends in a segment of
  no length. On the last segment the change is not seen at all, which
  changes nothing drawn. Reproduction: the path (0, 0), (1, 0), (2, 0)
  with the step 2 where x ≥ 1 and 1 elsewhere grades to [step 1: (0, 0),
  (1, 0), (1, 0)], [step 2: (1, 0), (2, 0)]
  (`isophotes::tests::a_change_of_width_on_a_vertex_doubles_it_as_swift_does`).
  The zero-length segment draws nothing, so no reference differs.

## 2026-11-16 · Phase 4: parametric surfaces, what came across

Ported whole, a module per Swift file: Parametric.swift's behaviour into
`parametric.rs` (the round torus in its angles and isothermal, the surface
displaced along its normal with its lifted tubular neighbourhood, the linear
curves off `VectorField::constant`'s flow, the fold curves and the grid's
normals and folds), Visibility.swift into `visibility.rs` (footprint,
front-most test, sheet, `onto_crease`, fold crossing, `HiddenLine::clip_on`),
SurfaceImage.swift's rasterizer into `surface_image.rs`, Cover.swift into
`cover.rs` (`PeriodLattice`), LinesOfCurvature.swift into
`lines_of_curvature.rs`, and Scene.swift's parametric branch of `visible`
(kurven-core); Torus.swift into kurven-landscape's `torus.rs`
(`PeriodicTorus`, `Drawing`, `PeriodicOptions` for its defaulted
arguments); and all of KurvenDynamics into kurven-surfaces (`forced.rs`,
`drawing.rs` with `SurfaceRequestExt` for Swift's extension of
`SurfaceRequest`, `bundle.rs`). The bake renders a parametric scene's
coordinate image and traces its outline; `bake` reads a surface's bundle
and `--dump` writes strokes and frame; the CLI has `surface`.

## 2026-11-16 · The coordinate image is Swift's CPU rasterizer, not Metal's pass

*Superseded 2026-11-16, its last paragraph: sn in relief at 0.4 is past its reach and tested by the image, not sphere traced; see "sn in relief past its reach: the coordinate image, not the port".*

Swift's bake takes a parametric surface's coordinate image from Metal
(`MetalRenderer.renderSurface`, `kv_param_vertex` in Shaders.swift). What
Metal computes: the grid's points uploaded as float32 texels; view and clip
matrices in float32; two triangles a cell, corners (00, 10, 01) and
(10, 11, 01), cells in rows of u, a periodic axis's index wrapped;
coverage by Metal's rasterization rules (a sample exactly on an edge
belongs to one triangle, by the top-left convention); a depth pass that
MAX-blends view z interpolated in float32; and a second pass for the
coordinates, kept by a LESS depth test on view z normalized over the
surface's padded range into a depth32Float attachment, so of two
fragments equal in that depth the first drawn (primitive order) keeps the
pixel, and the coordinate (lo + spacing · index, unwrapped) interpolated
affinely, as under an orthographic camera perspective correction is.

The port reproduces none of the float32 and takes Swift's own CPU
rasterizer instead, `SurfaceImage.rasterize`, the reference Swift's suite
holds the Metal pass to ("metal: the GPU coordinate pass is the CPU
rasterizer's": coverage and depth differ only on edges, under half a
percent of pixels). Its choices, all f64 and deterministic: the same
triangles in the same order, sampled at the frame's grid points (where
`metal_clip` puts pixel centres); covered where all three barycentric
weights are at least -1e-12, so an edge sample is claimed by both
triangles; depth and coordinate the weights' combination of the corners';
the depth kept as a float32 and replaced only by a strictly greater one,
so of equal depths the first triangle drawn keeps the pixel, which is
Metal's primitive-order rule; one depth for both, where Metal decides the
coordinate by its own normalized depth; a triangle of doubled pixel area
within 1e-12 of zero, or with a corner that is not a number (where Swift's
`Int` traps), draws nothing. Tiled bakes stitch the sub-frames as Swift
does. How it is held: Surfaces.swift's "the rasterized coordinates land
where their depth and pixel say" and "visibility by coordinate agrees
with the exact ray-cast" pass at Swift's tolerances; the golden
`surface/raster.txt` pins its depth and coordinate bits and
`surface/torus-bare` a bake tested by it, both identical native and wasm.
No reference drawing reads it: sn, sn in relief, the torus and periodic
all have a tubular neighbourhood and are sphere traced (Swift too), so the
float32 differences cannot move a reference stroke; only a surface with no
closed-form neighbourhood, the forced torus, is tested by the image, and
no reference drawing is one.

## 2026-11-16 · The surface reference drawings: every layer at 0

`references` bakes sn, sn in relief (0.4), the torus with a 2/5 linear
curve, and `periodic` sn(z, 0.5) over its periods, each by `surface ...
--save B; bake B`. Against Swift's: the same layers, the same stroke
counts in every layer (sn 20/77/14/24/6; sn-relief 21/79/10/17/3/8;
torus-linear 73/4/6; periodic 20/78/13/24/6), Hausdorff distance 0 and
relative length 0 at the SVG's four decimals in every layer. They are held
to the landscapes' floor, 1.5e-4 page units and 1e-5 of length. The
frame's two gallery drawings (`gallery-frame-edges`, `gallery-frame-turing`)
are `surface solid --kind frame`, a polyhedron: Phase 5's, left out of the
list with a PORT comment, as are the solids and the composite.

## 2026-11-16 · Phase 4: differences from Swift, by construction

- **Preconditions are values.** `ParametricSurface::torus`, `torus_over`,
  `displaced`, `grid_fold_curves`, `interpolate_normal` answer a
  `ParametricError` (`NotATorus`, `NormalCount`, `NoVolume`) where Swift's
  preconditions trap; `is_front_most` and `clip_on` answer `NoImage` and
  `ClipOnError`; `linear_curves` asserts its preconditions as
  `ParamAxis::new` does (and a step count past 2^63, where `Int` traps).
  `SurfaceDrawing::periodic` on a torus or a forced request panics, as
  Swift's `preconditionFailure`.
- **Loops for `concurrentPerform`.** The fold curves' rows, cuts and
  seeds, the displaced surface's normal offsets, the forced torus's values,
  positions and slices, the traced lines of curvature, `clip_on`'s vertices
  and cuts: loops in order, each iteration its own slot.
- **NaN.** The fold tracer bins a point that is not a number in no cell
  (Swift's `Int` traps). The forced fit's step counts are refused as
  `DynamicsError` where `Int(x.rounded())` would trap; an ODE failure is
  `DynamicsError::Ode` (Swift rethrows the integrator's error).
- **The periodic drawing asks for the refiner's hook once.** Swift reads
  `refiner.refine` per layer, finding the singular points each time; the
  port finds them on the first layer that asks and shares the hook. The
  points and so the curves are the same.
- **The bake.** A landscape's or a solid's outline is refused as Phase 7's
  (the depth pass); `--dump`'s frame for them is `DepthFrame::covering`
  of the view bounds at the resolution, what Swift's depth frame is,
  computed without a pass. A parametric bake renders the image with or
  without an exact test, as Swift's Metal bake does, and passes it on.
- **The CLI.** `surface solid` is refused (Phase 5), and so is a preview
  (`--preview`, a .png output; Phase 7); `--isophotes` and `--hatching` on
  a surface parse and are refused when drawn (Isophotes.swift, Phase 5).
  Stdout prints the same facts in the port's spellings.

## 2026-11-16 · Bake times and the wasm module (Phase 4)

On this machine (Apple Silicon), single-threaded, from the request to the
strokes: `surface sn` (grid 1024, sn sampled at 600, a 3000-pixel image)
0.51 s native through `kurven-cli surface sn -o` and 0.63 s in the wasm
module under Node 26 (`bake_surface_reference`, `bench.mjs`); `surface sn
--relief 0.4` (grid 8192) 1.97 s and 2.69 s. Both include the CPU
rasterizer's 3000² image, which sphere tracing then does not read. The
wasm module is 1,729,812 bytes (from 1,594,979), with the surfaces and the
golden generator's surface set linked in.

## 2026-11-16 · Phase 4: the suites, at Swift's tolerances

In `crates/kurven-surfaces/tests`, one Rust test per Swift suite, none
loosened, none skipped to get green:

- `surfaces.rs`, Surfaces.swift's surfaceTests, surfaceBakeTests and
  foldTests: 11 pass. Visibility by coordinate and by sphere tracing: 0 of
  20,000 disagree with the exact ray cast in all eight cases; the displaced
  torus's sphere tracing agrees with the walk up the line of sight; the
  clipped circles and the parametric bake are within 0.000% of the ray
  cast's length; `surface sn` bakes the same SVG at 1000 and 3000; fold
  curves within 1.5 px of the ray cast's outline.
- `periodic.rs`, creaseTests, periodicTorusTests and linearCurveTests: 6
  pass (the truncation edge drawn once, seen where either face is, hidden
  past the reach only by a sheet in front; every line on the sn torus ends
  on a zero, a pole or its continuation, none loose; a double zero and pole
  hold two rays of every level). The quarter periods were already
  kurven-math's.
- `conformal.rs`: Conformal.swift's five pass (the right angle at 778
  crossings to 4.9e-16 rad).
- `lines_of_curvature.rs`: LinesOfCurvature.swift's eight pass (about 80 s,
  nearly all the forced torus's fit and its 48 lines).
- `forced.rs`: Forced.swift's six pass (the preview's grid folds within
  0.13 px of the exact ones). Its orbit suite bounds Swift's 25 ms frame on
  the grid folds alone, the CPU half; the frame's GPU half is ignored.
- `paint.rs`: Paint.swift's two pass. `bundles.rs`: Bundles.swift's four
  parametric-surface suites and the surface half of "contract: bundles
  decode into consistent values" (all six surface bundles) pass.

Ignored, each for the GPU (Phase 7): "metal: the GPU coordinate pass is the
CPU rasterizer's", "preview: a parametric surface draws what its bake
draws", "preview: a torus nearly edge-on keeps its outline whole at rest",
and the GPU frame of the orbit suite; the drawing suite's last check
(a warm renderer's preview against a fresh one's) is left out for the same
reason. The four tests Phase 3 left for Phase 4 pass: "faces: a parametric
surface is one face", "fields: the contours of u and v on the torus are its
coordinate curves", the round torus's half of the occlusion suite, and the
torus's half of "integral curves: a linear curve is the constant field's".
No suite turned up a porting bug.

## 2026-11-16 · Phase 4 status

**Ported**: everything the handoff's Phase 4 names and every `// PORT:
phase 4` hole; none is left in the code. The new refusals are other
phases': a coordinate hatching or isophotes on a parametric surface
(Isophotes.swift, `PORT: phase 5`), `surface solid` (Phase 5), the outline
of a landscape or a solid and the CLI's preview (`PORT: phase 7`).

**The gate**: check-web green at the branch's head, with the merge of main
(PR #67) in. Four surface reference drawings at 0 (sn, sn-relief,
torus-linear, periodic), 15 drawings held in all; the frame's two gallery
drawings wait for Phase 5. Goldens: `surface/` (nine files), native and
wasm byte-identical.

## 2026-11-16 · Review of PR #69: degenerate frames, tiling, and what the bake renders

- **Tiles past the rows.** A parametric bake with more passes than pixel
  rows split into empty or one-row sub-frames: Swift's clip divides by a
  sub-frame axis of no length, and Metal cannot make a texture of no rows.
  The port takes no more passes than the frame has pairs of rows and of
  columns (`--resolution 2 --tiles 3` bakes in one), which changes no
  stroke, since the passes are stitched; `Bake::tiles` reports the passes
  taken.
- **Flat and one-sample axes.** `SurfaceImage::rasterize` divided by the
  axis's length and its samples less one, so a frame with a view-bound axis
  of no length, or a single row, came out blank. Such an axis's samples all
  lie at its lo; the port rasterizes there once, in view units, and copies
  the line to every sample, drawing a triangle seen edge-on along the other
  axis from the front-most point of its edges. Swift has no such case; a
  regular frame is rasterized as before, to the bit.
- **The image only where it is read.** Swift's bake renders a parametric
  scene's coordinate image always. A sphere-traced surface whose clipped
  layers all carry coordinates, with no outline asked for, reads none of
  it, so the port renders none (`bake_with_image` does, for what tests the
  image itself) and reports the frame from the view bounds; the image is
  shared by the layers' visibilities rather than copied, and a parametric
  surface's fold curves are placed once per line of sight. The strokes, the
  references and the goldens are unchanged.

## 2026-11-16 · Where the Swift looks wrong (Phase 4, ported as it is)

- **A run of two is carried by its end.** `PeriodLattice.canonical` brings
  a run into the rectangle by the translation of its middle vertex,
  `run[run.count / 2]`, which for a run of two vertices is its last. A run
  `clipped` to end on the far edge has that end in the next cell, so the run
  is carried a whole period too far, off the rectangle. Reproduction: the
  cell [0, 4) × [0, 2) and the run (3.5, 1), (4, 1) come out as (−0.5, 1),
  (0, 1); three vertices ending there, (3.25, 1), (3.5, 1), (4, 1), stay as
  they are. On the torus the two copies are one curve, so nothing draws in
  the wrong place, but its coordinates are a period off what the doc
  promises. Ported as it is; `cover.rs` pins it.
- **A steep linear curve's v runs on.** `linearCurves` cuts a path every
  eight turns of u and brings both coordinates back then, but never cuts
  for v, so its doc's "never more than eight turns from zero" holds for u
  alone: at a slope of 50 and a bit, over 24 turns, v runs about 400
  periods (2500 radians) between seams. The preview reads the coordinates
  as float32, whose step there is about 2.4e-4 rad, far coarser than a
  pixel of a fine grid's coordinate. Ported as it is; `parametric.rs`
  pins it.

## 2026-11-16 · Second review of PR #69: the fold memo, the pass, the dump

- **The fold memo is each scene's own.** It was an `Arc` shared by every
  clone of a scene and keyed by the line of sight alone, so a clone
  drawing other creases from the same camera could read another scene's
  folds. A clone now starts with an empty memo; a test draws sn in relief
  with and without its truncation edge from one camera.
- **A parametric bake renders the frame whole.** Swift tiles its Metal
  pass past the texture limit and stitches the tiles; the CPU rasterizer
  has no limit, and renders the frame once, which is what the stitched
  tiles stand for. A tiled request bakes the untiled image to the bit
  (`Bake::tiles` reports one pass). A surface's coordinate hatching, which
  carries its coordinates when it is tested, no longer counts as a layer
  that needs the image.
- **`--dump` names a layer's files by its place.** Swift writes
  `PREFIX.NAME.npy` and `.idx.npy`, so two layers of one name, or a layer
  named "outline" beside the traced outline, write over each other. The
  port writes `PREFIX.INDEX.NAME.npy`, INDEX the layer's place in the
  frame's `layers`: a deliberate difference.
- `Composite::clip_on_chords` shares `HiddenLine::clip_on`'s bisection and
  runs (`clip_on_runs`), placing the cut on the chord; the `surface`
  command reads `--lines-of-curvature` once.

## 2026-11-16 · Phase 6: the session, what came across

`Sources/KurvenApp`'s state and edit logic is `crates/kurven-session`, a
module per Swift file, without the views: Document.swift (`document.rs`: the
state, opening a bundle, adopting a landscape, a surface or a polyhedron with
the camera fitted or kept, navigation, levels, `drawn_layers`, the bake of
the scene the window shows), Edits.swift (`edits.rs`: `Edited`, `edit`,
`set_edited`, a drag as one step, `refusal`, undo and redo), Landscape.swift
(`landscape.rs`: `create`, `landscape_edited`, `pump`, `restyle`, `redraw`,
`set_truncation`, hatch counts, the plateaus' axis, `save_bundle`,
`Styling`, `applying(overrides:)`), Surfaces.swift (`surfaces.rs`),
Polyhedra.swift (`polyhedra.rs`), Paint.swift (`paint.rs`: the layers on a
face, the Add menu, add, remove, painting a function, the paint sliders),
Picking.swift (`picking.rs`), the five sliders' callbacks (`sliders.rs`:
`LandscapeSlider`, `LevelSlider`, `StrokeSlider`, `LightSlider`,
`SurfaceFieldSlider`, and the truncation slider of LandscapeSection.swift),
and App.swift's headless command line (`headless.rs`: `headlessPaint`,
`headlessBundle`, `headlessSurface`, `headlessLandscape`, `headlessPick`,
`resample`), which `kurven-cli app` runs. Two additions: `frames.rs`, the
draft and rest tiers of a frame (MetalView.swift and `LineGeometry.layOut`'s
engine half), and `protocol.rs`, the session as JSON messages and typed
arrays, which kurven-wasm exports.

Not ported, being views or the GPU's: App.swift's windows, menus and
panels, Inspector, the three sections' SwiftUI, Gallery and Thumbnails (the
picker's pictures are drawn by the preview), FrameClock, MetalView. A
headless `--screenshot`, `--inspector` or `--thumbnails` is refused as
needing the GPU (Phase 7), and a command line that would leave a window open
is refused.

To read a bundle in memory, where the page has no file system, the three
bundle readers read from a `Source` (a directory, or `BundleFiles` under a
name) and give the files they write (`files()`); the path readers and writers
are those over a directory.

## 2026-11-16 · Swift's detached tasks are jobs, landed in the order they started

The document's slow work in Swift -- a sampling, a restyle, a layer's levels,
a surface's build or restyle, a polyhedron's restyle, a face's boundary --
runs on a detached task whose result lands back on the main actor, where it
cannot interleave with an edit. Here each is a `Job` queued on the document
(`jobs.rs`) holding what the task captured, and `settle` runs them in the
order they were started, landing each as Swift's continuation does; a
landing can start more. Swift's headless run waits for the document to
settle after every edit (`settle()` in `headlessPaint`), and so does every
request of the protocol.

The one way this can differ: two of Swift's tasks in flight at once land in
whichever order they finish. One edit starts two at once only where it changes a
landscape's styling and its samples together, as an undo across steps can,
and there the restyle, from grids in memory, finishes first in Swift as it
lands first here. No gate script shows a difference. Swift's status lines carry the time
a job took; the port's do not (`Instant` panics on wasm32-unknown-unknown,
and a status is not a drawing).

## 2026-11-16 · A pick reads a CPU cast until the page has a depth pass

Swift picks a face at a pixel by reading the preview's depth texture
(`MetalRenderer.previewDepth`) and taking the texel's centre back through the
camera. Until Phase 7's depth pass, `picking::preview_depth` casts the
texel's centre against the triangles the depth pass draws, in doubles: a
landscape's grid decimated by its step at the drawn height (min(|f|, h), or
the clamped grid under bands), per tile and cut to its region as
`kv_height_vertex` cuts it, with the occluder's triangles; a parametric
surface's grid; a solid's `mesh()`. Held to Swift on 42 pixels of each of
the seven gate inputs and sn in relief: all 336 picks name the same face (or
nothing) as Swift's app, and the points agree to the four decimals Swift
prints. `Document::pick(point)` takes the world point, as Swift's
`pick(at:)` does, so the page can pass the one its own depth pass reads.

## 2026-11-16 · The binding is the plain C ABI, not wasm-bindgen

The handoff left the choice to Phase 6. The C ABI is kept:

- **Size.** The C ABI adds no JS glue and no runtime: the exports are plain
  functions. wasm-bindgen adds its generated JS and its describe machinery,
  which its CLI then strips from the module. It was not measured: neither
  the crate nor its CLI is on this machine, and the CLI is a tool, which the
  rules say to ask for.
- **Simplicity.** The protocol is narrow on purpose: a JSON request in, a
  JSON reply and some typed arrays out. That is `kv_alloc`, `kv_free`,
  `kv_stage`, `kv_call` and four readers of the reply, 82 lines of
  TypeScript (`web/app/src/engine/module.ts`) and no build step. With
  wasm-bindgen the module could not be loaded without the generated glue,
  whose version must match the crate's, and the goldens' Node harness would
  need it too.
- **Typed arrays.** Equal. Either way an array leaves linear memory by one
  copy (wasm-bindgen's `Vec<f32>` return copies into a `Float32Array`; here
  `memory.buffer.slice`), and that copy is what the Worker transfers to the
  page without copying again. Neither can transfer linear memory itself.

The golden sets keep their C-ABI path, behind a default feature (`goldens`);
the page's module is built `--no-default-features`.

## 2026-11-16 · The session against Swift's app: 28 edit scripts

`crates/fixtures/swift/session/scripts.txt` holds 28 of the app's headless
command lines; `make.sh` runs Swift's KurvenApp (`swift build -c release`)
over them, on eight input bundles Swift's kurven-cli wrote (gamma at 160,
gamma banded at 140, (z²−1)/(z²+1) at 120, sn, sn in relief at 0.4, a torus,
the frame under Turing's pattern, a cube), and keeps what each saved and
baked: 32 manifests and 27 drawings (8.2 MB with the inputs). Every flag the
gate names is in them -- `--set` (a surface's field, a landscape's window
and resolution), `--levels`, `--strokes`, `--relief`, `--truncation`,
`--pick` with `--add` and `--remove`, `--function` with and without
`--periods`, `--undo` -- and `--redo`, `--axis`, `--hide`, the catalog's
`--surface` and `--landscape`, and both of App.swift's paths (in order,
`headlessPaint`; levels then strokes then fields, `headlessBundle`).
check-web's session step replays each line through `kurven-cli app`. Result:
all 32 manifests byte for byte Swift's; all 27 drawings with Swift's layers,
Swift's stroke counts in every layer, and Hausdorff distance 0 and length 0
at the SVG's four decimals in every layer, held at the reference drawings'
floor, 1.5e-4 page units and 1e-5 of length. The wasm module under Node
writes every file the native run writes byte for byte (`session.mjs`, in the
identity step), the resampled landscapes' grids among them.

## 2026-11-16 · sn in relief at 0.5 and over: a Phase 4 difference

The first relief-undo script, `--relief 0.6 --relief 0.2 --undo`, failed: its
mag_minor layer is 9.4e-3 page units from Swift's. The session is not the
cause: `kurven-cli surface sn --relief 0.6 -o x.svg` differs the same way
between the two CLIs, and the session's own drawing is the Rust CLI's to the
bit. By amplitude, the CLIs against each other: 0.1, 0.2, 0.3 (and the
reference 0.4) at 0; 0.5 layer 3 at 2.0e-2; 0.6 layer 1 at 9.4e-3; 0.7
layers 0, 1, 3 up to 4.2e-2; 0.8 layer 0 at 1.5e-2; 0.9 layers 1 and 5 up to
7.5e-2; 1.0 layers 1 and 5 up to 9.6e-2. At 0.6 the stroke counts agree and
the difference is a visibility cut: a stroke's end 2.5e-3 apart and a
two-vertex stub placed at another point of one curve. Phase 4 held sn in
relief at 0.4 only. That lane is Phase 4's and was not opened here; the gate's
relief edits stay under 0.5 and say why. Reproduction: `kurven-cli surface sn
--relief 0.6 -o a.svg` in each, `kurven-cli compare`.

## 2026-11-16 · Where the Swift looks wrong (Phase 6, ported as it is)

- **A level count, then a resample, saves one thing and draws another.**
  `gamma.kurven --hide folds --levels mag_major=5 --set res=100 --save
  s.kurven --bake d.svg`: the bundle saved has mag_major at 5 levels, the
  drawing has it at the manifest's 9 (25 strokes, where the same two edits
  in the other order draw 14). The resample's landing adopts the new bundle
  with `curvesWanted = nil`, and nothing asks for the counted levels again;
  `levelCounts` still holds the count, which `drawnLayers` saves. The port
  does the same (gate script `gamma-levels-then-window`, at 0 against
  Swift). The fix in both: adopt a resampled bundle with `curvesWanted =
  false` when `levelCounts` is not empty, as `install` does after a restyle.
- **`--open` cannot be used with an edit.** `headlessPaint` walks every
  argument and refuses `--open` as "not an editing step", so the paint path
  takes the bundle only as a bare argument, which App.swift's own message
  tells you not to use for a window. Ported as is; the gate passes the bare
  path.
- **`--set` edits a landscape only on the paint path.** `headlessBundle`'s
  `--set` reads a surface's fields alone ("gamma has no field 'res'; it has:
  "), where `headlessPaint`'s edits a landscape's window too. So `--set
  res=100 --save s.kurven` fails and `--hide folds --set res=100 --save
  s.kurven` works. Ported as is.

## 2026-11-16 · Phase 6: differences from Swift, by construction

- **No file system.** A run of the app's command line reads its bundle
  through an opener and returns what it saves, bakes and dumps;
  `kurven-cli app` writes them.
- **Statuses without times**, above; and `--drag truncation|levels N`
  reports no main-actor stall, which a Worker does not have.
- **The accent colour** of a picked face's boundary is macOS's default blue,
  `#007aff`; Swift reads the system's.
- **Undo.** Swift's `UndoManager`, grouped one registration an event, is a
  stack of steps, each putting back the `Edited` before it and filing the
  opposite step, as the closure `registerUndo` files does; a new step clears
  what could be redone.
- **Numbers on the command line** are read by Rust's parser (no C
  hexadecimal floats), as `kurven-cli` reads them.

## 2026-11-16 · The Worker protocol

`web/app/src/engine/`: `protocol.ts` (the requests, the document's state,
the messages both ways), `module.ts` (the C ABI), `worker.ts` (`serve(port)`,
one request at a time, each reply's arrays transferred) with
`worker-entry.ts` for a browser Worker, and `client.ts` (`EngineClient`).
Requests are answered in order, numbered by the client. A view (a frame, a
bake, the grids) is asked for under a key: at most one per key is in flight,
a newer ask waits replacing any waiting one, and the in-flight one is stale
when it lands if a newer ask came meanwhile; a stale ask's promise resolves
`{ stale: true }`, its data dropped. A hung request is cancelled by hand or
by a timeout: the client terminates the Worker, rejects what is pending with
`Cancelled`, starts a new Worker, and replays the journal (every bundle
staged and every edit answered) before any new request, so the new Worker
holds the document as it was before the request that hung. The TypeScript
uses erasable syntax only (`erasableSyntaxOnly`), so Node 26 runs it as it
is; `web/app/test/worker.test.mjs` (check-web, "the Worker protocol under
Node") drives the client and the module in a worker_threads Worker: open,
truncate, save (Swift's manifest byte for byte) and bake (native's SVG byte
for byte), undo, three asks of one view (stale, stale, delivered), and
`besseli(1e12, z)` painted on a torus cancelled by a 1.5 s timeout, after
which the document is the one before the edit and takes the next edit.

## 2026-11-16 · Interactive timings: one thread, native and wasm

cn and gamma from the catalog at the app's default resolution (600), through
the protocol (`kurven-session/tests/interactive.rs`, ignored, and
`kurven-wasm/interactive.mjs`), on this machine (Apple Silicon), one thread,
wasm the check's module (opt-level 3) under Node 26:

| request | cn native | cn wasm | gamma native | gamma wasm |
|---|---|---|---|---|
| open: sample, contours on f | 566 ms | 855 ms | 140 ms | 346 ms |
| camera change, draft frame (folds on the grid) | 10.5 ms | 20 ms | 3.4 ms | 5.6 ms |
| camera change, rest frame (the window's layout) | 1,783 ms | 2,664 ms | 250 ms | 690 ms |
| camera change, judged strokes (the bake) | 23.1 s | 34.9 s | 1.02 s | 2.93 s |
| truncation drag, a step (restyle) | 267 ms | 349 ms | 88 ms | 213 ms |
| truncation drag, draft frame | 10.4 ms | 16 ms | 3.5 ms | 5.4 ms |
| truncation drag, rest frame (the window's layout) | 1,537 ms | 2,287 ms | 237 ms | 673 ms |
| truncation drag, judged strokes (the bake) | 18.4 s | 27.8 s | 1.01 s | 2.90 s |

The rest frame is Swift's window's (`LineGeometry.layOut(draft: false)`): the
folds on the exact fold and tested against the solid, and the clipped curves
on the walls tested by their wall; every other curve is left to the GPU's
depth test. Swift's was ~127 ms on cn on 18 cores; the port's single thread
takes 1.78 s native, 2.66 s in wasm, of which the folds are 1.27 s (911
vertices) and the walls 0.53 s (5,356 vertices) natively; gamma's 134 and
107 ms. The judged strokes, every curve tested as the bake tests it at a
quarter pixel, are 23 s on cn: not a rest frame anyone waits for.

What would close the gap, measured or not:

1. **At rest, draw what Swift's window draws**, not the judged strokes: the
   page's rest frame is `frame` with `draft: false` and the GPU's depth test
   (Phase 7); the judged strokes are what Export writes, made on demand and
   cancellable. This is the handoff's "judged strokes at rest" corrected by
   the numbers.
2. **A pool of Workers without threads.** Swift spreads this work over its
   cores with `concurrentPerform` (the fold curves' rows and seeds, each
   tested vertex), and the port's loops are the same pieces in order, so it
   splits across N Workers, each holding the same document (the journal
   replays into each, as a restart does), each answering for its share of
   the rows and curves. No cross-origin isolation is needed; the cost is N copies of the
   document and of every sampling. At 8 Workers cn's 2.66 s would be about
   0.35 s, gamma's 0.69 s about 0.1 s. Not built.
3. **Wasm threads** (rayon over `SharedArrayBuffer`) need COOP/COEP
   headers, which GitHub Pages cannot set (a service worker can fake them
   for a same-origin page); the hosting question (handoff question 3)
   decides it. Not adopted, as the brief says.

## 2026-11-16 · The wasm module's size, by profile

Each the `web` profile (LTO, one codegen unit, panic=abort, strip) with an
opt-level, built in a target directory of its own; time from the same
interactive script under Node:

| build | bytes | gzip -9 | cn rest | gamma rest | cn bake | gamma open |
|---|---|---|---|---|---|---|
| opt 3, with goldens (the check's) | 2,511,593 | 864,444 | 2,679 ms | 703 ms | 34.9 s | 339 ms |
| opt "s", with goldens | 2,017,870 | 695,015 | 3,000 ms | 855 ms | 38.8 s | 432 ms |
| opt "z", with goldens | 1,729,302 | 599,127 | 3,127 ms | 922 ms | 40.7 s | 470 ms |
| opt 3, no LTO, 16 units | 2,687,260 | 911,700 | 2,691 ms | 700 ms | 34.6 s | 342 ms |
| opt 3, the page's (no goldens) | 2,220,820 | 756,264 | | | | |
| opt "s", the page's | 1,772,180 | 611,088 | | | | |
| opt "z", the page's | 1,518,175 | 523,797 | | | | |

"s" saves 20% of the bytes for 11 to 27% more time, "z" 31% for 17 to 39%
(its draft frames are twice as slow); LTO saves 7% and no time. The golden
sets are 290 KB of the module. The profile stays at opt-level 3: the page's
module is 756 KB gzipped, and the engine is slow enough already. No
binaryen (`wasm-opt`): it is a tool to ask for.

## 2026-11-16 · Phase 6 status

**Ported**: Sources/KurvenApp's state and edits (`kurven-session`), the app's
headless command line (`kurven-cli app`), the session as a protocol, the
wasm binding (C ABI) and the Worker protocol with its client. **The gate**:
28 edit scripts against Swift's app, 32 manifests byte for byte and 27
drawings at 0 (check-web, "the session's edit scripts"); the same scripts
byte-identical through the wasm module under Node (the identity step); the
Worker protocol end to end under Node; the session's own tests
(`kurven-session/tests/document.rs`: an edit one undo step, a drag one step
and a draft until let go, a refusal no step, the git sha an undo puts back).
**Not done**: the relief lane above (Phase 4's); the GPU's half of the
frames, the depth pass a pick reads and the picker's pictures (Phase 7); the
page (Phase 8). Nothing of the session waits on a `PORT:` hole.

## 2026-11-16 · Review of PR #70: the client, the frames, the protocol

- **Restarts.** A second cancel during a replay took the half-rebuilt
  journal and lost what was not yet replayed, and the first cancel's
  promise rejected unhandled. Now a replay leaves the journal whole until
  it has run, a restart terminates the Worker being replayed into and
  replays the same journal into the next, an overtaken boot is rejected
  (a terminated Worker never says it will not load) so its replay gives
  way, and every caller of an earlier restart waits for the last replay.
  Tested with cancels back to back, one 40 ms into a replay, and one exactly
  at a replay's second request (against a Worker that fails on purpose,
  which then receives all four journaled requests).
- **close()** rejects what is pending, clears its timeouts and refuses
  every later call; a hung edit pending at close no longer restarts a
  Worker when its timeout comes.
- **Failures.** A stage has an id and its own reply, so a bundle the
  engine refuses rejects that stage, which is not journaled; a failure
  says whether the module trapped, and only a trap restarts. An engine
  that will not load is said to the request waiting, and the next request
  loads it again.
- **Journal by effect.** Every reply says whether the request touched the
  document (`touched`: false only for a refusal of its arguments, or an
  open of a name never staged). A failed open, which forgets the
  document's edits and leaves it a failure, is journaled and replays to the
  same failure.
- **Compaction.** From the last open, create or createSurface (each a
  clean slate), keeping the last viewport, mode, draft resolution and
  refine before it, which an open keeps, and every staged bundle, which a
  later open may read; a run of gestures as one `camera` request, set to
  the bit from the state's `camera.exact` (gestures move the navigator
  alone, and summing them would not be exact: the orbit's folds do not
  associate in floating point, and its elevation is clamped); a run of
  moves of the landscape, truncation, light or levels slider as its last
  (each sets a value outright and is never refused; a surface field's or a
  hatching's step can be refused and keep the one before, so theirs are
  kept); a run of one outright setting as its last; a bundle staged again
  over an earlier one that no open between reads. A 300-gesture session
  with a five-move drag and twenty viewports replays as its three staged
  bundles and its open plus five entries, to the same document and camera to
  the bit. What is not compacted: a toggle (`hide`) and a refusable move.
  The replayed document equals the last one reported but for status lines,
  which are messages (a save's, a bake's, a pick's are not replayed).
  One exception to "to the bit": JSON cannot write -0.0, so a camera angle
  that is exactly -0 would replay as +0.
- **Frames** drew `framed_scene`; they draw `shown_scene` now, the picked
  face's boundary last, each layer saying whether it is drawn
  (`shown_visible_layers`, Swift's `shownOptions`).
- **Paths in memory** were found by exact key where a directory joins the
  path; both read `normal_path` now (`./a.npy`, `a//b.npy`), and a file in
  memory is borrowed, not copied, as are the staged files an `open` or a
  headless run reads. `call` parses a request once.
- **Indices and sizes** (hide, preset, viewport, the resolutions) are
  whole numbers or refused, where `as usize` made -1 and 1.5 into 0 and 1.
- **The page's module** (`--no-default-features`) now runs the session's
  scripts byte-identically to native and the whole Worker test, in
  check-web, as the check's module does.

The session gate is unchanged: 32 manifests byte for byte, 27 drawings at
0, against the committed Swift outputs.

## 2026-11-16 · Second review of PR #70

- **A failed open is no clean slate.** Swift's `Document.load` sets
  `.loading`, clears the scene, the navigator, what was queued and the
  edits, and on a read that fails sets `.failed`; the landscape's or the
  surface's request, the level counts, the hidden layers and the styling
  stay. The engine does the same (it is Swift's), and the request it kept
  shows in the state. The client's compaction cut the journal at the
  failed open and replayed a document without that request; it now cuts
  only at an open that succeeded. Tested: open gamma, open a broken
  bundle, set rMax, drag rMax, cancel; the replayed document is the live
  one (the test fails with the old rule). The edits after a failed open do
  nothing in either: `edited` is nil, and a landscape drag's draft finds
  the samples asked for already "shown" (`pump`'s `shown.samples(as:)`),
  so nothing is sampled. That is Swift's, kept.
- **A request racing a restart** waited for the `ready` it read, and could
  be posted to the new Worker before it had loaded and been replayed. It
  waits again while a restart has replaced the Worker, checked by its
  epoch as it is sent.
- **A journaled entry that traps on every replay** restarted the Worker
  forever. Two traps in a row on one entry cut the journal there, with
  what follows (which may depend on it), and the next call is told
  (`JournalTruncated`).
- **Unknown layers.** Swift's `paintLevels`, `paintStrokes`, `paintLight`
  and `paintResetLevels` pass over a layer they do not find; its headless
  run refuses one before ("no layer ... with levels to drag"). The protocol
  refuses one as the headless run does, touching nothing, so it is not
  journaled.
- **save** says why it cannot save (`save_bundle` returns the reason).
- **Copies.** A reply's array is encoded once from where its values lie
  and moved into the module's reply; `kv_stage` takes the staged file's
  allocation (`kv_alloc` lends a boxed slice) rather than copying it, so a
  staged bundle is held once; the copy the host makes out of linear memory
  is the one the Worker transfers.
- **BundleFiles** is a type of its own, its paths normal when put in, so a
  lookup is a plain get. `kurven-cli app` reads a bundle's regular files
  alone, follows no symbolic link and reads no dotfile.

## 2026-11-16 · sn in relief past its reach: the coordinate image, not the port

Phase 6 found `surface sn --relief A` drawn differently by the two CLIs from
A = 0.5. Reproduced here, the CLIs against each other at the reference
floor (1.5e-4 page units, 1e-5 of length):

| A | layers off | Hausdorff | length | strokes |
|---|---|---|---|---|
| 0.4 | none | 0 | 0 | the same |
| 0.5 | 3 (ang_minor) | 1.995e-2 | 7.7e-4 | the same |
| 0.6 | 1 (mag_minor) | 9.44e-3 | 3.9e-4 | the same |
| 0.8 | 0 (mag_major) | 1.53e-2 | 5.0e-4 | the same |
| 1.0 | 1 (mag_minor), 5 (folds) | 9.67e-3, 9.64e-2 | 6.0e-4, 3.8e-4 | 81 against 78, 12 against 13 |

Each difference is a stroke's end moved by a vertex, or a two- or
three-vertex stub drawn by one side only, nearly all on the vertical
through the torus's axis.

**The first divergence is the coordinate image.** With R = 2 and r = 1 the
relief's reach is 1 and |sn| is truncated at 3, so the relief bounds a
solid, and is sphere traced, only while 3A < 1. From A = 1/3 it has no
tubular neighbourhood and every vertex is decided by the front-most test
on the coordinate image (Visibility.swift's `isFrontMost`), which Swift's
bake renders by Metal and the port by the CPU rasterizer. Two probes, one
each way:

- Swift's KurvenCore, KurvenDynamics and KurvenBake in a scratch package,
  baking each amplitude once as `bake` does (Metal; byte for byte the CLI)
  and once with `SurfaceImage.rasterize`'s image in Metal's place: the
  rasterized bake is the port's SVG byte for byte at 0.4, 0.5, 0.6, 0.8
  and 1.0.
- The port in a scratch crate, given the Metal image the Swift probe
  dumped (its float32 depth and coordinates): it bakes Swift's CLI SVG
  byte for byte at all five.

So no stage of the port's surface, contours, folds, visibility or clip
differs from Swift's. The earlier entry "The coordinate image is Swift's
CPU rasterizer, not Metal's pass" says no reference drawing reads the
image, sn in relief being sphere traced: wrong for sn-relief, whose 0.4 is
past the reach (the suite's own `in_relief_past_its_reach_...` says so).
It matched Metal at 0.4 by luck: two vertices change verdict there, each
alone between hidden neighbours, so no stroke moves.

**Why the images differ there.** sn's poles lie on the inner equator, at
u = 0 and u = 2K with v = K', and the plateau round each, |sn| = 3, stands
at x = ±(1 − 3A): from A = 1/3 each crosses the axis into the relief rising
from the other side of the ring, and the two sheets pass through each
other. Of the 7.0 million pixels both images cover, 15 to 28 hold
different sheets at the five amplitudes, all within 0.03 page units of the
axis's vertical up to 0.8 (within 0.38 at 1), and 2 to 13 are covered by
one only. At those pixels the two sheets' exact
depths are 1.7e-5 to 2.3e-4 apart, the order of the grid's chord error,
and the images' depths differ by 1e-6 to 5e-6: Metal interpolates in
float32 under its own rasterization rules, the rasterizer in doubles with
inclusive edges, and each keeps the sheet its chords put in front. Of the
24 vertices whose verdict changes over the five amplitudes, 15 have a pixel
holding the other sheet among the nine the test reads, one a pixel covered
by one image only, and 8 a difference in the footprints those pixels read.

**Which is right: neither, decidably.** An exact cast (Newton on the map
from every grid cell whose projected box holds the vertex; hidden when
another sheet, more than two cells off in coordinates, meets the line of
sight nearer by over 1e-9 of scale; checked against sphere tracing at 0.3,
where it hides no vertex sphere tracing sees) hides all 24. The rasterizer
agrees with it at 15, Metal at 9: which sheet a raster keeps where two
sheets tie in depth is the image's resolution, as Visibility.swift's doc
says of relief ("a raster is right only to its resolution, and relief puts
the curves where that runs out"). Reproducing Metal's pass to the bit
(fixed-point snapping, float32 interpolation, its depth normalization) is
not something the port can do from Apple's documentation, and the Phase 4
decision stands: the port's image is Swift's rasterizer.

**The gate.** `references` gains sn-relief-1, `surface sn --relief 1 --save
B; bake B` with the coordinate image by `SurfaceImage.rasterize` in place
of Metal's (`surfaceRasterized`, an additive change to References.swift;
scripts/check.sh green with it; every other reference baked byte for byte
as committed). The port's drawing is Swift's byte for byte: every layer at
0, strokes 23/78/11/20/5/13, held at the floor. Swift's Metal bake of the
same is not a reference, for the reason above, and references.txt says so.
The test `sn_in_relief_past_a_third_is_tested_by_the_image_and_passes_through_its_axis`
(kurven-surfaces, periodic.rs) pins where the relief stops bounding a solid
(0.33 is sphere traced, 0.34 is not) and the plateaus at x = ±(1 − 3A).

**For the owner.** Swift's Metal-baked sn in relief past a third of its
range draws a few strokes' ends by which sheet a float32 pass keeps where
the relief passes through itself; the session gate's relief edits (PR #70)
at and above 0.34 are tested by the image too, and its 0.4 matches by luck
as `sn-relief` does. Both images also keep the same vertices an exact cast
hides, about 250 of 34,000 at 0.6, many of them back sheets just behind a
fold whose front part the front-most test's tolerance (two footprints and
a cell) takes for the vertex's own neighbourhood: the blind spot
Visibility.swift's doc names, which facing covers only on a surface that
bounds a solid. The port and Swift agree on every one of them.

## 2026-11-16 · Review of PR #71: one rasterized path, the how, the test

- **Every image-tested surface reference is rasterized, by one path.**
  References.swift's `baked` draws a parametric surface with no tubular
  neighbourhood by `rasterizedBake` (Bake.swift's frame, tolerance and
  layers, with the coordinate image by `SurfaceImage.rasterize` in place of
  Metal's; written once, a named mirror of Bake.swift, since sharing the
  code would mean editing KurvenBake). So sn-relief at 0.4 is now baked
  so too, and comes out byte for byte as its Metal bake did; no SVG of the
  32 changed. The surfaces table in both CLIs carries the flag, which adds
  ", its coordinate image by SurfaceImage.rasterize in place of Metal's"
  to the `how`; Swift refuses a scene whose flag is wrong. The header says
  so. `references` takes 10.1 s, as before.
- **The test** reads sn's poles at iK' and 2K + iK' from the quarter
  periods, each plateau against its own foot on the bare torus, and checks
  the reach at 0.3 and 0.4 rather than at 0.33 and 0.34 (the entry above
  names the old pair).

## 2026-11-17 · The slow suites on four cores, checking as much

CI's native-test step took 17.5 minutes on ubuntu-latest's 4 cores, 724 s
of it composite.rs. Each slow suite was profiled (macOS `sample`) and sped
up on its reference side or by running independent work side by side. No
sample point, case, tolerance, assertion or count changed, and no engine
code: every change is under `crates/*/tests`.

**Measuring at 4 threads.** The suites spread their oracles over
`available_parallelism()`. `KURVEN_TEST_THREADS` now sets that number
(`common::threads`, in both crates' test commons), and `RUST_TEST_THREADS`
sets libtest's. Timings below run each test binary as cargo does, in its
package directory, on this 18-core Mac: "4 threads" is both set to 4,
"full" neither. 4 threads on 18 cores is not 4 cores, so CPU time is given
too. From the CI run, CI's wall time was 0.5x this machine's CPU seconds
for composite.rs and 2-3x this machine's 4-thread wall time for the rest.

| suite | before, 4 threads (wall / CPU) | after, 4 threads | before, full | after, full |
| --- | --- | --- | --- | --- |
| landscape composite | 109.2 s / 1413 s | 5.5 s / 37 s | 88.6 s / 1541 s | 5.1 s / 40 s |
| landscape walls | 16.7 s / 51 s | 10.3 s / 58 s | 7.2 s / 61 s | 5.7 s / 67 s |
| surfaces forced | 18.6 s / 47 s | 8.4 s / 51 s | 18.7 s / 48 s | 8.1 s / 54 s |
| surfaces lines_of_curvature | 19.9 s / 22 s | 7.4 s / 22 s | 19.1 s / 21 s | 4.6 s / 25 s |
| surfaces bundles | 18.2 s / 20 s | 7.6 s / 21 s | 18.1 s / 20 s | 7.4 s / 22 s |
| landscape stress | 10.0 s / 32 s | 5.0 s / 26 s | 10.4 s / 37 s | 5.3 s / 29 s |
| every native test binary | 218.7 s / 1625 s | 64.9 s / 257 s | 182.0 s / 1772 s | 56.0 s / 281 s |

By the calibration above that is about 3 minutes on CI. Walls' CPU is the
same single-threaded (47 s before and after); the extra CPU seconds at 4
and 18 threads are this machine's efficiency cores taking more of the work.

**Where the time went, and what changed.**

- **composite.rs**: over 95% in `first_hit`, the brute force over every
  triangle of the landscape's drawn mesh, five times a sample (the cast and
  its four nudges), over about 11 million samples in the two gate tests. `RayIndex` asks the same
  Möller–Trumbore test, the same operations on the same values, of only the
  triangles an origin can meet. Each triangle is projected along the line
  of sight and bucketed on a grid over its box, widened by a bound on how
  far its 1e-12 barycentric allowance and the rounding of the test can
  carry a hit outside it. Nearly edge-on triangles are asked from every
  origin, and far origins against every triangle. So the set of hits is
  `first_hit`'s, and the least hit is the same double. The gate's casts and
  `TerrainTops` reach it through `RayCache`, built along the first
  direction asked.
- **stress.rs**: the real drawing's grid cast. `exactly_hidden` is
  common's alone now, generic over `Triangles` (a mesh, or its
  `MeshIndex`). It asks `under_some_triangle` through a `FootprintIndex`.
  Straight up, that tests the triangles whose footprint box holds the
  point, widened by the 1e-12 an edge's half-plane is let through by. Along
  the line of sight, it tests those whose footprint spans the line's offset
  across it and reaches as far along it as the point. That cast went from
  1.74 s to 13 ms. The bake suite's steep erf case is built and drawn on a
  thread of its own beside rgamma's, its two drawings side by side and its
  layers measured each on its own, summed in order. The walk over a
  reflected or turned tile stops once its least gap is under f by more
  than the ten-thousandth the comparison asks, because the least gap only
  falls.
- **par_map** handed each thread one contiguous chunk, so the cameras near
  edge-on, which come last and walk farthest, all fell to one thread. Each
  thread now takes the next items as it comes free, and the results come
  back in order.
- **walls.rs**: half `Heightfield::visible_curves` on f (the engine under
  test), half `SolidWalk::depth` evaluating |f|. Both are fixed costs. Only
  the sharing-out above changed. Its remaining 47 CPU seconds are the floor
  short of fewer samples.
- **forced.rs, lines_of_curvature.rs, bundles.rs**: engine work run in
  sequence, independent item by item: Fourier-torus jets, exact folds,
  tracings, builds and bakes. These now run side by side, and every check
  is made in its order with its message. This covers forced.rs's nineteen
  request edits (after the timed fit and re-place), the preview's four
  exact-fold cases, and the orbit's seven exact folds (after its grid folds
  are timed in turn as before). It also covers each line-of-curvature
  tracing, four seeds at a time (`traced_integral_curves` maps each seed on
  its own, and the curves are joined in the seeds' order), and each bundle
  request, whose direct bake now runs beside its round trip. The 1024 x 512
  forced torus is built once for forced.rs, as the fitted torus already was.

**The proof that nothing checks less.** `KURVEN_RAY_INDEX_CHECK=1` makes
every `RayIndex` and `FootprintIndex` answer be checked against
`first_hit` and `under_some_triangle` themselves. composite.rs, stress.rs,
contact.rs and isophotes.rs pass with it set (composite.rs in 93 s on 18
cores). Every test binary of the workspace was run one test at a time with
`--nocapture`, at main (bb2af7a) and here. With durations masked, the
printed output is the same in all 51 binaries. composite.rs's eight
compositions show 0 wrong in the same counts, for example "a frame sunk
into the slope of a tower: 0 of 1390687 disagree, 647314 hidden by another
... own: 0 of 657338 disagree", and only the order in which they finish
differs. stress.rs's real drawing shows "grid: 14384 vertices, traversal
285, cast 285, pit 1032/48/48". lines_of_curvature.rs shows "59965 chords
... 48 curves, 48 returned ... 609 crossings". walls.rs and stress.rs now
print what they tally, at main too for the comparison: for example, "Γ at
0.5 | 1 | 1.5 | 2.5: 0 of 61110 kept under the solid", "reflected: 0 of
4560 disagree, 3820 clear", and "steep: 82.08779809582452 of
82.08779809582452 units kept; 507 behind, 2078 buried".

**Left as it is.** walls.rs's riser walks are the slowest left, at about
12 s of CPU per core at 4 threads. They cannot get much faster without
fewer samples, or a bound on |f|'s slope that the walk does not have. If
CI needs more, the riser suites are the candidates for a slower lane.

## 2026-11-17 · Review of PR #74: timed stretches, meaning, cleanup

- **Timed stretches run alone.** forced.rs (the orbit's grid folds under
  25 ms, the re-place under a quarter of the fit), stress.rs (rgamma's two
  drawings under 60 s) and lines_of_curvature.rs (the arc-length refusal
  under 2 s) assert wall-clock times. Each timed stretch holds `alone()`,
  and every stretch of the same file that spreads over the cores with
  `par_map` holds `sharing()` (`crates/test-support/threads.rs`, a lock per
  test binary), so no saturating work runs while one is timed.
  Single-threaded tests still run beside it, as on main. The cost at 4
  threads, by libtest's clock: forced.rs 8.4 s to 12.9 s, stress.rs 5.0 s
  to 9.9 s.
- **The steep case's two drawings are made one after the other** again,
  so it checks, as on main, that a drawing made after the first is the
  same.
- **The tracing times mean something.** Each of lines_of_curvature.rs's
  tracings (both families, with returns, then once round) is one
  `par_map`. Each is timed as the wall time around it, and the message says
  so. A call takes seeds / threads seeds, not four, so the face and its
  principal directions are built 8 times at 4 threads, not 48.
- **Cleanup.** threads() and par_map live once, in `crates/test-support`,
  which each crate's tests/common includes by `#[path]`. RayCache keeps an
  index per direction asked, so no direction falls back to the brute force
  unseen. TerrainTops' mesh lives in its RayCache alone. RayIndex and
  FootprintIndex share `Reach`, the ball an origin must lie in for their
  widening to hold. The bundle suite bakes from the scene and moves the
  drawing, rather than cloning it.
- **Checked as before.** The printed output of every test binary is main's
  with durations masked; only the tracing message's added words differ.
  `KURVEN_RAY_INDEX_CHECK=1` passes on composite, stress, contact and
  isophotes.
- **Timing note.** /usr/bin/time's `real` counts the time this laptop
  sleeps, and two runs here read 953 s and 1070 s for suites libtest timed
  at 10 s. libtest's own "finished in" is the figure to read.

## 2026-11-17 · Phase 7: the preview, what came across

`Sources/KurvenMetal` is split where its class mixes two jobs. What a frame
draws is decided in the engine, in doubles: `kurven-session/src/preview.rs`
ports `SceneResources` and `SurfaceResources` (`Resources`, keyed on
`Scene.content`), `LineGeometry.isTestedExactly` and `layOut`
(`is_tested_exactly`, `lay_out`), `SurfaceStrokeGeometry.layout`, and
`renderPreview`'s and `renderParametricPreview`'s choice for each layer of
pipeline, depth tolerance, colour and width, with the uniforms
(`KVUniforms`, the frame-wide `KVShading` and `KVStroke`, the coordinate
pass's padded depth range). The protocol's `preview` request sends it as
JSON and float32 arrays. Drawing it is the page's: `web/app/src/gpu`:

- `shaders.ts`: the WGSL port of Shaders.swift, one module as Swift's one
  library: the heightfield's implicit mesh and the walls into the depth
  pass, MAX-blended into r32float; a parametric surface's depth pass and
  coordinate pass (rg32float behind a LESS test on normalized view depth);
  paper, shaded (normal from screen-space derivatives, no depth
  attachment) and the depth view; strokes as screen-space quads tested at
  the centreline against the depth pass with the slope-scaled tolerance,
  hidden where a graph's normal faces away; surface strokes tested by the
  front-most coordinate and facing; and `kv_layout_probe`.
- `layout.ts`: the structs at WGSL's offsets, and `probeLayout`, which has
  the GPU read every field back (Swift's layout probe).
- `renderer.ts`: `PreviewRenderer`, Swift's passes in Swift's order; the
  scene's resources a single-entry memo on the content, each layer's
  segments a buffer under the engine's key, so a camera change uploads the
  uniforms and the layers the camera moved and nothing else.
- `view.ts`: the canvas, drafts while the camera moves and the frame at rest
  a quarter second after (MetalView's clock), one frame asked at a time,
  the newest wish waiting; `fallback.ts`: without WebGPU, the judged
  strokes drawn with Canvas 2D at rest.
- `main.ts`: the least page that shows it: a canvas, drag to orbit,
  shift-drag to pan, the wheel to zoom toward the cursor, `f` to fit, `1` `2`
  `3` for the modes, as Swift's GestureView maps them; the query picks a
  catalog landscape or surface. The formula field and the rest are Phase 8's.

## 2026-11-17 · Phase 7: differences from Swift, by construction

- **No infinity in a shader.** Swift hands Metal an infinite depth
  tolerance (drawn as it comes) and an infinite truncation height (none).
  WGSL lets an implementation assume no infinity reaches a shader, and
  canonical JSON cannot write one, so the engine sends `null` and the page
  writes the largest float, `UNREACHABLE`; `stroke_visible` passes a stroke
  at that tolerance without adding to it, and `coord_rate`'s starting
  "infinitely far" is the same constant.
- **WGSL's layout, not C's.** A `vec3f` is twelve bytes in WGSL and sixteen
  in MSL, so `KVShading.ambient` sits at 44, not 48; the probe holds the
  page's writers to the shader's reading of every field of every struct,
  and passes on SwiftShader and on Metal.
- **A buffer per layer.** Swift keeps every layer's segments end to end in
  one buffer and passes each draw its `first`; the page keeps each layer's
  under the engine's key, so a frame that moves the folds re-uploads the
  folds alone, and `first` is always zero and is not passed. A layer's
  `KVShading` and `KVStroke`, set per draw by `setVertexBytes` in Swift,
  are a `KVLayer` slot per layer in one uniform buffer.
- **`float32-blendable`.** The depth pass MAX-blends an r32float target,
  which WebGPU allows only with that feature. SwiftShader and Metal both
  have it; an adapter without it gets the fallback (judged strokes at
  rest), not a second depth semantic.
- **The rest frame blocks the Worker.** Swift lays out the tested strokes
  on another thread and draws drafts meanwhile; the engine is one Worker,
  so a draft asked during cn's 2.2 s rest layout waits for it. The interim
  is Phase 6's list ("Interactive timings"): a pool of Workers, or wasm
  threads.
- **`frame`'s rest layout is now `layOut`'s for every scene.** Phase 6's
  `rest_layout` tested a landscape's folds and walls and left a surface's
  and a solid's layers as carried; it is now `preview::lay_out`, Swift's
  `layOut(draft: false)`: a sphere-traced surface's folds and clipped
  curves with coordinates, and a solid's clipped layers, as the bake tests
  them.
- **A pick still casts on the CPU.** Swift reads the preview's depth
  texture; the page's is on the other side of the Worker, and a pick needs
  the point only to name the face. `preview_depth` keeps Phase 6's cast,
  its triangles now `depth_pass::drawn_triangles`, the one enumeration the
  CPU depth pass and the cast share. One change from Phase 6's: a landscape
  whose decimated grid has fewer than two nodes a side still has its
  occluder's triangles cast, as Metal draws the walls whatever the grid.
- **No preview natively.** `kurven-cli surface --preview` (a PNG), and the
  app's `--screenshot` and `--thumbnails`, stay refused, now saying the
  preview is the page's: a native build has no GPU, and the port's preview
  is WebGPU in the page.

## 2026-11-17 · Headless Chromium and WebGPU: what works here

Playwright 1.63.0 (`@playwright/test`, a dev dependency of web/app, the one
added) pins Chromium revision 1243, Chrome for Testing 153.0.8010.12, which
was already in ~/Library/Caches/ms-playwright. Probed on this machine
(Apple Silicon, macOS), over http://localhost (WebGPU wants a secure
context; `about:blank` has no `navigator.gpu`):

| browser | flags | adapter | a WebGPU canvas |
|---|---|---|---|
| headless shell (old headless) | none | none | -- |
| headless shell | `--enable-unsafe-webgpu` | SwiftShader | device lost at the first frame: "A valid external Instance reference no longer exists" |
| Chrome for Testing, new headless (`channel: "chromium"`) | `--enable-unsafe-webgpu` | Apple, metal-3 | draws |
| the same | `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader` | Google, SwiftShader (fallback adapter) | draws |
| the same | `--disable-gpu` | none (`navigator.gpu` is there, `requestAdapter` answers null) | -- |

So the lane runs Chrome for Testing in its new headless mode, never the
shell: `swiftshader` (Dawn's CPU Vulkan, the same on any machine),
`gpu` (the machine's GPU, Metal here) and `fallback` (`--disable-gpu`, the
page with no adapter). Both adapters offer `float32-blendable`; SwiftShader's
largest texture is 8192 (Metal's 16384), which the renderer asks for and
checks. `--disable-features=WebGPU` and `--disable-webgpu` leave the Metal
adapter in place; `--disable-gpu` is what takes WebGPU away.

## 2026-11-17 · The browser lane's gate: the preview at rest against the judged strokes

`scripts/check-browser.sh` builds the page's module and the page, puts the
module beside it, and runs `web/app/test/browser` against `vite preview`.
The measure is Swift's own, from "preview: a parametric surface draws what
its bake draws" (`test/browser/coverage.js`): a preview pixel is a stroke
where 299 r + 587 g + 114 b < 224,000; the judged strokes (the engine's
`bake` of the same frame, page points being view x and y) are rasterized
along each centreline at a quarter pixel into the frame's pixels
(`DepthFrame::index`); recall is the share of judged pixels with a preview
stroke within one pixel, precision the share of preview stroke pixels with
a judged pixel within two, both held above 0.99 with more than 5,000 judged
pixels, at a 900 by 700 viewport, the slope allowance off (the bake's
predicate), each scene at the camera it opens with. Results, identical in
the coverage on both GPUs but for a few drawn pixels:

| scene | judged px | recall | drawn px (SwiftShader / Metal) | precision | control: unclipped px stray |
|---|---|---|---|---|---|
| gamma (catalog, landscape) | 32,683 | 100.00% / 99.99% | 54,494 / 52,749 | 100.00% | 14.2% of 58,138 |
| cn (catalog, landscape) | 52,612 | 99.97% | 73,283 / 72,805 | 100.00% | 3.4% of 74,090 |
| sn (catalog, surface) | 34,587 | 100.00% | 39,342 / 39,401 | 100.00% | 27.4% of 60,780 |
| the picture frame (polyhedron) | 31,545 | 100.00% | 41,615 / 41,548 | 100.00% | 29.7% of 61,343 |

**The control.** Swift draws the same curves with no visibility test and
asks that over a fifth of their pixels have no judged stroke within two
pixels, or the measure proves nothing; on its torus that holds, as it does
on sn (27.4%) and the frame (29.7%). On a landscape it does not: the
hidden contours of gamma and cn run within two pixels of visible ones far
more often (14.2%, 3.4%). The lane asks the control of every scene in the
form the gate itself gives it -- drawn unclipped, the curves fail the gate's
precision, stray > 1 − 0.99 -- and Swift's fifth of sn and the frame. That
is a weaker control on cn than Swift's on its torus: recall carries the
landscapes' gate, and a preview drawing cn's hidden contours would show at
96.6% precision, under the 99%. No threshold of the gate was changed.

**Without WebGPU** (`fallback`, Chromium with `--disable-gpu`: no adapter),
the page says why ("WebGPU has no adapter here") and draws the judged
strokes with Canvas 2D at rest; held to their own rasterization by the same
measure, all four scenes are at 100.00% recall and precision. The bake is
the slow part there: cn's judged strokes take 35 to 46 s in Chromium's
Worker (Phase 6's 34.9 s under Node), and the page shows nothing new for
that long.

## 2026-11-17 · Frame times in Chromium: cn and sn

From the lane's timing test: open from the catalog (the app's resolution,
600), at 900 by 700, eight orbit steps of (6, 1) pixels each drawn as a
draft, then the frame at rest, three times over; medians. "Engine" is the
`preview` request answered in the Worker (wasm, one thread), arrays and
all; "GPU" is from taking the reply in to the queue's work done.

| frame | Metal: engine | Metal: GPU | Metal: total | SwiftShader: GPU | SwiftShader: total |
|---|---|---|---|---|---|
| cn, draft | 16.1 ms | 2.6 ms | 18.9 ms | 1,440 ms | 1,456 ms |
| cn, rest | 2,214 ms | 7.0 ms | 2,221 ms | 1,394 ms | 3,613 ms |
| sn, draft | 1.6 ms | 2.8 ms | 4.3 ms | 665 ms | 667 ms |
| sn, rest | 90 ms | 4.9 ms | 95 ms | 416 ms | 506 ms |

A draft costs the engine cn's fold curves traced on the grid for the camera
(Phase 6 measured 20 ms under Node) and sn's none -- its grid folds are
derived for each camera too, from the grid normals the engine keeps -- and
uploads the uniforms and the folds' segments alone (the rest are held under
their keys: 6.9 MB on the GPU for cn, 4.8 MB for sn). On a real GPU the
passes take 3 to 7 ms; on SwiftShader, a CPU, seconds, which is why the
lane's numbers for it are coverage, not speed. cn's rest frame is the
engine's 2.2 s of tested folds and walls (Phase 6: 2.66 s under Node),
which a camera that keeps moving waits behind (above, "the rest frame
blocks the Worker"); Swift's window does it in ~127 ms on 18 cores. sn at
rest sphere traces its folds and every clipped curve: 90 ms.

## 2026-11-17 · The depth pass of a landscape or a solid, on the CPU

Phase 4 left the bake's outline of a landscape or a solid refused, waiting
for a depth pass. The evidence on what reads Swift's depth pass, from
Bake.swift and the app: the outline (`DepthImage.outline`), the `depth`
output, a pick (`previewDepth`), and the preview's per-fragment test. No
stroke of a bake reads it. So the preview's pass is the page's GPU, and the
bake's is the CPU's: `kurven_core::depth_pass` rasterizes the triangles the
pass draws (`drawn_triangles`: the decimated grid at the drawn height,
tile by tile, cut to the region as `kv_height_vertex` cuts it, then the
occluder; a solid's mesh; a parametric surface's grid) into a
`DepthImage` with `SurfaceImage::rasterize`'s choices (the frame's grid
points as samples, inclusive edges at -1e-12, float32 depth replaced only by
a strictly greater one). `bake` renders it, whole, when the outline is
asked for, and traces the outline from it; the CLI reports the pass
(`bake gamma.kurven --outline 0.5`: 4000 by 4000, 0.257 s against 0.188 s
without). Picking
keeps its per-pixel cast, now over the same `drawn_triangles`. Held by a
cube seen from above (every sample inside its square covered at its top, to
the float, none outside) and the bake's own test (the outline is the last
layer, black at the width asked, inside the frame, and the same however the
pass is asked to be split). Not compared against Swift's Metal pass: no
Swift golden carries an outline, and the two differ on edges by
construction, as the coordinate image's do.

Two halves of Swift suites in `kurven-landscape/tests` that read a depth
buffer were left out for the want of one ("stress: on a real drawing the
traversal is the brute-force ray cast"'s count of what a 1000-pixel buffer
hides in the pit; "solids: ... the depth pass rasterized its faces"). They
can be restored with `depth_pass::render_depth`; those files were being
reworked by another agent during this phase and were not touched.

## 2026-11-17 · Swift's GPU suites in the browser lane

Phase 4 left four suites ignored for the GPU. They cannot run natively; the
stubs in kurven-surfaces now say where each runs, and three run in
`web/app/test/browser/surfaces.spec.ts`, each scene a surface bundle (the
session fixtures' manifest with Swift's request and camera put in), at
Swift's thresholds:

- **"preview: a parametric surface draws what its bake draws"**: the round
  torus at 512 by 256, 24 by 12 coordinate curves and its folds, shear 0.5,
  azimuth 25°, elevation -50°, 900 by 700. At rest and in a draft, recall
  and precision 100.00% (19,380 judged pixels; 36,497 drawn) on Metal and
  on SwiftShader; the control 35.2% stray (Swift asks over 20%).
- **"preview: a torus nearly edge-on keeps its outline whole at rest"**: sn
  (sn's grid 200, the surface's 600) from xAngle -89.5°, zAngle 118.2°, at
  1400 by 900, the folds alone: at rest 100.000% of 4,213 judged pixels
  (Swift asks over 99.9%), the draft 99.525% on Metal and 99.644% on
  SwiftShader (Swift asks under the rest less 0.2%): the draft, tested per
  fragment against the coordinate image, breaks the outline as in Swift.
- **"metal: the GPU coordinate pass is the CPU rasterizer's"**: the WebGPU
  depth and coordinate passes against `SurfaceImage::rasterize` over the
  same frame (the protocol's `surfaceImage` request: the CPU image, with the
  view, the buffer-order clip and the padded depth range the GPU renders it
  by), on Swift's two tori (R/r 2 and 1.25, 384 by 192) at 800 pixels from
  two cameras, one sheared (bundle presets, where Swift builds its cameras
  in the test; any camera holds the claim). On Metal: 26 to 38 of about
  500,000 covered pixels differ in coverage or by more than 1e-4 of the
  depth span, every one on an edge (Swift: all on edges, under half a
  percent); depth elsewhere within 5.5e-4 absolute; and where the two
  agree, the GPU's coordinate is the CPU's front-most point's within 0.02
  of a grid cell everywhere off an edge (in place of Swift's check against
  the map, which the CPU image passes natively). **On SwiftShader the claim
  fails, and is skipped there, not loosened**: 7,486 pixels differ, 5,102
  of them off an edge, nearly all in depth alone, by 1e-4 to 3e-3 of the
  span, half each way, at about 1.5% of the covered pixels; Metal has 22
  such. SwiftShader's interpolation of a varying across a triangle is
  coarser than Metal's; the strokes, which read the coordinate image and
  not its depth, pass the two suites above on it.
- **"surfaces: an orbit frame of the forced torus, folds and all, keeps
  up"**, the GPU half: not ported. A frame is now the engine's draft in the
  Worker and the page's passes, both timed by the lane on cn and sn; the
  forced torus's fit takes about 80 s natively, past what the lane waits
  for. The CPU half (the grid folds, median under 25 ms) passes natively.
  The drawing suite's last check (a warm renderer's preview against a fresh
  one's) is not ported either.

## 2026-11-17 · The browser lane in CI

`.github/workflows/check-browser.yml` runs `scripts/check-browser.sh
--project swiftshader --project fallback` on ubuntu-latest, after `npx
playwright install --with-deps chromium`. It has not run: nothing has been
pushed. On Linux, Dawn reaches SwiftShader through Vulkan, which Chromium
leaves off unless asked, so the config adds `--enable-features=Vulkan
--disable-vulkan-surface` there and nowhere else; whether that suffices on a
runner is the workflow's first finding. The `fallback` project needs no GPU
at all, and the `gpu` project is for a machine with one (the script runs it
on a Mac). A run takes about five minutes here, most of it cn's judged
strokes (about 35 to 46 s each in the Worker) and SwiftShader's frames.

## 2026-11-17 · Phase 7 status

**Ported**: the preview. The engine half of KurvenMetal
(`kurven-session/src/preview.rs`: resources keyed on content, the layout of
each layer under a key, the per-layer choices and the uniforms) and the
protocol's `preview` request; the WGSL shaders, the renderer, the three
modes, the layout probe and the canvas in `web/app/src/gpu`; the fallback
without WebGPU; the least page that shows it (`main.ts`). The CPU depth pass
of a landscape or a solid (`kurven_core::depth_pass`) and with it the bake's
outline, which was Phase 7's refusal; picking keeps its cast over the same
triangles. Every `PORT: phase 7` hole is closed or, where it needs a GPU the
engine does not have (the CLI's preview PNG, the app's `--screenshot` and
`--thumbnails`), a refusal that says the GPU is the page's.

**The gate** (`scripts/check-browser.sh`, Chromium 153 by Playwright 1.63.0):
gamma, cn, sn and the picture frame at rest against the judged strokes,
recall 99.97 to 100.00% and precision 100.00% on SwiftShader and on Metal,
Swift's thresholds; the fallback drawing them at 100.00%; the WGSL structs
read back as written; three of Swift's four GPU suites, one of them on Metal
only (above). 22 passed and 8 skipped by design (each project's own) on
this machine. check-web green, without a browser; the session's 28 scripts
and the goldens unchanged.

**Not done**: the browser lane has not run on Linux or in CI; the forced
torus's GPU frame suite; the two landscape suites' depth-buffer halves (in
files another agent was reworking); the rest frame still blocks the Worker
(cn 2.2 s); the page is Phase 8's.

## 2026-11-17 · Review of PR #73: keys, the queue, the outline's memory

- **Keys across a restart.** `ContentID`s are counted per process, and a
  restarted Worker counts from one again, so a key the page held could name
  a scene of the new engine, which would then send nothing for it. Every
  preview key now starts with the engine's nonce, which the Worker gives at
  load (`seed`, 16 hex digits from `crypto.getRandomValues`; empty
  natively), and the page lets go of what it holds when the client
  restarts. Held by the Worker test (a restart, the same scene replayed and
  a new one: no key of the new engine is one the old gave) and a native
  test of two seeded sessions.
- **What a layout's key depends on.** The camera key hashed the scene's
  resolution, which every zoom changes, so a zoom re-sent sphere-traced and
  solid layouts and draft folds. It is Swift's `cachedLines` key now, the
  camera's view and projection and whether a draft; a landscape's tested
  layout at rest adds the resolution it was placed to, and a held one
  placed at least as finely serves (`VisibleLines.serves`: zooming out
  keeps it, zooming in tests again). Tested on sn and gamma.
- **A layer the scene cannot test** (`Scene::visible` refusing) came back as
  carried with its tolerance already unreachable, drawing hidden curves.
  It is depth-tested now, as a draft leaves it, and the frame says why
  (`refusal`); the engine remembers the refusal by key, so a frame that
  sends no segments for it still tests it. No refusal is constructed in the
  port today, so this is untested by a real case.
- **One frame in flight.** `drawNow` went around the queue and could send a
  second `preview` with the same holdings; it goes through it now. A rest
  wished while a draft waited replaced it; the draft goes first now. A drag
  ends on `pointercancel` and `lostpointercapture` as on `pointerup`.
- **The outline's memory.** The bake rendered the whole frame's depth for
  the outline, about a gigabyte at 16000 pixels, then a coverage grid as
  large again. The pass is rendered 512 rows at a time now and only the
  coverage kept (`depth_pass::coverage`); every sample is the whole pass's,
  so the outline is the same (the depth pass test holds strips of 1 to 1000
  rows to the whole, and gamma's and the frame's outlined SVGs are the same
  bytes). The coverage grid the contour reads is still the frame's size in
  float32, as Swift's stitched buffer is.
- **One rule for the drawn grid.** `depth_pass::DrawnGrid` is the grid the
  pass draws (bands baked in, a uniform truncation the ceiling), shared by
  the GPU's resources, the CPU's pass and a pick, the bundle's grid shared
  rather than copied. Its step is at least one: Swift's Metal side passes
  `Heightfield.step` as it is, where every CPU reader in Swift takes
  `max(step, 1)`; a step of none would draw a flat sheet at one texel, and
  the port takes the CPU's rule everywhere. Mesh corners are looked up by
  one function (`for_each_triangle`), and the keys by one hash.

## 2026-11-17 · The browser lane in CI: the first run, and the second attempt

The first run of check-browser.yml on ubuntu passed the fallback and failed
every SwiftShader test within seconds: "WebGPU device lost (destroyed):
Device was destroyed", then "A valid external Instance reference no longer
exists" from `mapAsync` -- the old headless shell's symptom on the Mac, the
GPU process gone. The flags were `--enable-unsafe-webgpu
--use-webgpu-adapter=swiftshader --enable-features=Vulkan
--disable-vulkan-surface`. The second attempt, which cannot be run here:

- on Linux, `--enable-unsafe-swiftshader` (recent Chromium refuses WebGPU
  on SwiftShader without it), `--use-angle=swiftshader`,
  `--enable-features=Vulkan`, `--ignore-gpu-blocklist`, and Chromium's own
  warnings and errors on stderr (`--enable-logging=stderr --v=0`, printed
  under `DEBUG=pw:browser`); `--disable-vulkan-surface` dropped;
- a first test, `adapter.spec.ts`, that prints the adapter (vendor,
  architecture, fallback or not), its features and limits, whether it has
  `float32-blendable` (the depth pass's MAX blend, which the renderer
  checks before asking for a device, falling back with the reason), a
  device asked for, the layout probe, a canvas cleared and presented, and
  the device's loss with its reason; on this Mac it passes on SwiftShader
  and on Metal, with `float32-blendable` on both. The gate's failure says
  the page's fallback reason;
- the workflow runs the adapter test headless and headed under `xvfb-run`,
  then the SwiftShader project headless and, if that fails, headed under
  xvfb, and the fallback, and fails unless the fallback and one of the two
  SwiftShader runs pass.

If two rounds do not get SwiftShader working on ubuntu, the SwiftShader
project stays a local lane and CI runs the fallback alone.

Locally, one full run of the lane timed out on cn under `gpu` and
`fallback` and on the fallback's own test (600 s each, against 40 s and
70 s usually) while the machine stalled; the same tests re-run passed at
the numbers above (cn 99.97% / 100.00% on Metal, 100.00% in the fallback).

## 2026-11-17 · The browser lane in CI: WebGPU works on Linux, presenting does not

The second CI round answered it: on ubuntu, SwiftShader's adapter is there
(google/swiftshader, a fallback adapter, `float32-blendable` among its
features), a device is made, and the layout probe's compute pass passes;
the device is lost ("destroyed: Device was destroyed.") the moment a canvas
is presented, headless and headed under xvfb alike. The page never destroys
a device: it is the GPU process's canvas interop. (The adapter step itself
did not run: `--project swiftshader adapter` reads `adapter` as a second
project, `--project` taking a list; the filter goes first now.)

So the SwiftShader project draws **offscreen** (`?offscreen=1`,
`PreviewView`'s `offscreen`): each frame into a texture of the canvas's size
and of the format the renderer's colour pipelines are built for (the
preferred canvas format), read back by `copyTextureToBuffer`, the canvas
never configured. The pixels are the canvas's: the same renderer and frame,
pipelines, format and size, which `present.spec.ts` holds to byte for byte --
sn at rest drawn on the canvas and read back, then drawn again offscreen:
0 bytes differ, on Metal and on SwiftShader on this Mac, under each of the
three presentation projects. Presenting is its own project (`present`, the
swiftshader flags), apart from the gate, since on Linux it can take the GPU
process down, and twice more with the compositor elsewhere (`present-gl`:
`--disable-vulkan-surface --disable-features=Vulkan --use-angle=swiftshader`;
`present-angle`: `--use-gl=angle --use-angle=swiftshader`); on Linux each
is allowed to fail, saying how in its annotation. On the Mac's own GPU
(`gpu`) the gate presents, as before, and the presentation test must pass.

## 2026-11-17 · Phase 8: the page, steps 1 to 3

`web/app/src/main.ts` and `web/app/src/page/`: no framework, the engine
behind `EngineClient` as before, so a pool of Workers can replace the one
Worker without the page knowing. Every control sends an edit and shows the
state the reply carries; the page's only memory is where a hand is while
a control moves.

- **Step 1.** The formula field (`formula.ts`): return sends `expression`;
  a refusal carries `at` (below) and the field marks those characters with
  a mirror of its text laid over it in the same font, its left edge where
  the field's text puts them (held to 1.5 px by measuring the text). The
  catalog's dropdown, with "… (typed)" for a typed expression, as Swift's
  picker shows it. The canvas (`canvas.ts`): drag orbits, shift- or
  option-drag pans, the wheel or a pinch zooms toward the pointer, `f`
  fits; from the keyboard with the canvas focused, the arrows orbit,
  shift and the arrows pan, + and - zoom about the centre. Export SVG:
  `bake` with `svg`, downloaded, cancellable (a cancel restarts the Worker
  and replays the journal, as a hung request does).
- **Step 2.** The window as two thumbs on one track per axis, a field for
  each end, the ends a fiftieth of the reach apart, the reach the preset's
  window or room around a typed one (LandscapeSection.swift); Samples
  (120 to 2000 by 20); the truncation as None, At a height (a slider to
  max(3 h, 10) and a field) and Per band of Re (shown, its rows read only:
  bands are step 6); each layer's visibility, named in the glossary's words
  with its manifest name beside it, and its level count (2 to 80) with a
  reset; the camera's presets, Fit; Drawing, Shaded and Depth, and 1 2 3.
  ⌘Z and ⇧⌘Z (Ctrl on other machines) undo and redo.
- **Step 3.** The URL's fragment holds the state (`link.ts`): what is
  edited (`edited`: the landscape's request, its layers and truncation as
  drawn, the level counts), the camera to the bit (`camera.exact`), the
  hidden layers by name, the mode, the depth tolerance, the bake's
  resolution and whether contours are placed by f; JSON, deflated
  (`CompressionStream("deflate-raw")`, or plain where a browser has none),
  base64url, `#k1.`: about 1.3 KB for a typed expression. It is rewritten
  (`history.replaceState`) 0.4 s after the last change. A page opened on
  one sends `viewport`, `refine` and `mode` (what an open keeps), then
  `restore`, then `hide` by place, `depthTolerance`, `bakeResolution` and
  `camera`; the camera carries its framing, so a page of another size
  shows the same drawing. A link holds a landscape only: a surface, a
  polyhedron or a bundle opened read-only writes none (steps 5 to 7).

**Protocol additions** (kurven-session's `protocol.rs`, each tested in
`tests/protocol.rs`): `expression` refused with `at` = {message, position,
length}, in characters as the language counts them (an empty expression is
at 0, as language.json freezes it); `catalog` (the presets with their
windows); `edited`; `restore` (`Document::restore_landscape`: the request
sent with its styling as `landscape_edited` sends one, sampled and fitted
as `create` does, no edits to undo, the request then kept as linked -- a
framed adoption reads it back from the bundle, canonical expression and
all -- and the level counts set on the landscape adopted, since an
adoption clears them); `depthTolerance` (the Visibility section's field in
Inspector.swift, which a camera preset also sets). The client journals a
restore as an open and `depthTolerance` as a setting set outright, and
gives its journal's requests (`journaled`). `kurven-cli session REQUESTS
-o out.svg` runs requests natively, one a line, and bakes what they left.

## 2026-11-17 · Export is the native bake's bytes; a link is the drawing

`page.spec.ts` edits a document as a user does (a typed expression, a
window end, the truncation height, a level count, a hidden layer, the
shaded mode, an orbit and a zoom toward the cursor), clicks Export, and
compares the download to `kurven-cli session` running the page's journal
natively: the same bytes, in every project. Then the URL, opened in a
fresh page at 1180 by 760 (the first was 1000 by 800): `edited`, the
camera to the bit, every layer's state, the mode, the depth tolerance and
the bake's resolution are the first page's, its Export is the same bytes,
and the restored page's journal (which starts at its `restore`) run
natively gives them too. Native and wasm agreeing is the identity step's
claim; this adds that the page's Export is the judged strokes of the
document the page shows, and that a link loses nothing a drawing reads.

## 2026-11-17 · What a slow frame at rest forces

The Worker answers one request at a time and a wasm call cannot be
interrupted, so whatever the page asks while a frame at rest is being laid
out waits for it: a gesture's reply, the draft after it, an edit. Measured
by the lane (cn at 600, a drag during its frame at rest, to that drag's
draft drawn): 2.7 to 2.8 s on Metal, 5.5 s on SwiftShader (two runs each), which is cn's rest
layout (2.2 s) plus the draft's own time. The page adds no wait of its own:
gestures are summed while one is with the engine (one request behind,
never a queue); a slider's moves are coalesced, latest wins, and its
let-go waits only for the last move; a draft wished goes before a rest
waiting. The page itself never blocks (the engine is off the main thread):
fields take keys and controls move while the engine is busy, and the
indicator over the canvas says what holds it ("drawing at rest: testing the
folds and walls… 1.4 s", "sampling…", "exporting: baking the judged
strokes…") with the seconds. Cancelling a frame at rest to draw a draft
sooner would mean a restart and a replay (cn's open alone is 0.86 s of
sampling), so it is not done; a pool of Workers or wasm threads is the
owner's decision (Phase 6, "Interactive timings").

## 2026-11-17 · Where the Swift looks wrong (Phase 8, ported as it is)

- **A window or Samples drag samples nothing while it moves.**
  `LandscapeSlider.set` during a drag sets `document.landscape` and no
  more; `editingChanged(true)` samples the request as it was, at the draft
  resolution, and `editingChanged(false)` the new one in full. So the
  drawing shows a coarse copy of the old window for the whole drag, then
  the new window. The port does the same (probed: gamma's grid is the old
  window's at 240 after a move, the new one's at 600 after the let-go), and
  so does the page. A literal's slider does sample each move
  (`dragLandscape(draft:)` in `move`). The fix in both: `set` calls
  `dragLandscape(draft: true)` while a drag lasts.
- **A typed expression's window reach moves with the window.** Swift
  derives it from the current domain on every redraw, so it stretches under
  the thumb as it moves; the page pins it for the length of a drag, as
  Swift pins a literal's reach. A view's detail, not the engine's.
- **A slider's label is the request's key.** The protocol's landscape
  slider passes its label to `applying_overrides`, so the page names the
  window's and the resolution's sliders `rMin`, `rMax`, `iMin`, `iMax`,
  `res`, and the undo step is named so, where Swift's says "Re", "Im" and
  "Samples".
- **A hidden layer is baked.** Swift's Bake writes `framedScene` whole,
  hidden layers and all, and so does the port's: hiding is the preview's.
  The page's Export follows, and says so nowhere yet; the fallback, which
  draws the judged strokes, draws hidden layers too.

## 2026-11-17 · Phase 8 status (steps 1 to 3)

**Done**: steps 1 to 3 of the page (above), with five protocol additions,
`Document::restore_landscape`, and `kurven-cli session`. **The gate**:
`web/app/test/browser/page.spec.ts`, five tests run in the swiftshader
(offscreen), gpu (Metal) and fallback projects: the formula and its
error marks, at language.json's positions; the catalog, the canvas's
gestures (zoom's anchor held to 1e-9), a window thumb and the truncation
height dragged (drafts while the height moves, with WebGPU), a layer
hidden, levels stepped and dragged, the three modes, a preset, undo;
Export byte for byte `kurven-cli session` on the page's journal, the URL
to a fresh page of another size (the same document, the same bytes, and
natively from its journal); cn's frame at rest; phone width. The whole
lane: 40 passed, 11 skipped by design (each project's own; the slow-frame
test needs WebGPU), 7.9 minutes. check-web green, its gates unchanged
(44 goldens and 87 session files byte-identical native and wasm, the
session's 28 scripts against Swift's).

**Not done**: steps 4 to 7 (the picker's thumbnails, surfaces and their
controls, banded truncation's rows, picking and painting and undo's UI,
`.kurven` files); a link to a surface or a polyhedron; Swift's literal
sliders (a row for each number in the expression), its Perspective switch
and depth tolerance field, which the page has no control for (the depth
tolerance travels in the URL); the presentation and CI runs on Linux of
the new tests (not run here).

## 2026-11-17 · Review of PR #75: links held to the controls, gestures in order

- **A link is held to what the controls reach.** `restore` sampled
  whatever a link said: a resolution of a million or a billion levels would
  hang the Worker on open. It now refuses, touching nothing, a resolution
  outside the Samples slider's 120 to 2000, a window with no area, a level
  count outside a level slider's 2 to 80, a hatching of more than a strokes
  slider's 400 lines across the window, and a contour layer listing more
  than 1,000 levels (a bound on work; the catalog's own families list 40 at
  most). `bakeResolution` is refused as the bake refuses it (1 to 20,000),
  and `camera` refuses non-finite values and units per pixel or a distance
  that are not positive. Every catalog landscape still restores from what
  is edited. The page says a refused link in its notice line with the
  engine's reason, puts back the refinement and mode it sent first, and
  opens its own landscape, whose link then replaces the refused one.
- **Gestures are batched, not summed.** The entry above ("What a slow
  frame at rest forces") said gestures are summed while one is with the
  engine; summing put the camera elsewhere than the gestures one by one
  (the orbit's arithmetic does not associate, its elevation is clamped,
  zooms were taken about the latest anchor and reordered against pans).
  They now wait in order and go as one `gestures` request, which applies
  each as `gesture` does: one request behind, and the same camera to the
  bit, natively (a mixed sequence) and in the page (the gestures given
  during cn's frame at rest replayed one by one).
- **The page.** A link pasted into the tab is opened (`hashchange`) and not
  written over; the page's keys work wherever focus is but in a field
  that takes text (a select counts: it answers letters); a reply to the
  formula no longer overwrites or marks text typed after return, and text
  the engine replaces takes its marks with it; a second pointer on a held
  slider starts no second drag; a window's thumbs show the clamped value.
  The page tells edits from looks by the client's `VIEWS`.

Each has a test in `page.spec.ts` (the formula's checked to fail without
its fix). The lane: 55 passed, 11 skipped by design, 9.5 minutes; a draft
asked during cn's frame at rest drawn 2.8 s later on Metal, 5.6 s on
SwiftShader, as before. Export and the link's round trip are the same
bytes as natively, in all three projects. check-web green.

## 2026-11-17 · Second review of PR #75: a link is untrusted input

- **Every layer a link carries is bounded, or refused.** `linkable` is an
  allow-list: contours (at most 1,000 levels, finite), hatchings (at most a
  strokes slider's 400 lines across the window), wall edges (distinct
  sides of the window's four), the truncation edge, fold curves, slope
  lines (at most 400 seeds, listed or by arc length across the window);
  any other kind, a layer stored as arrays among them, is refused, as are
  more than 64 layers and a truncation that is not finite or has more than
  64 bands.
- **A refused link changes nothing.** `restore` makes the landscape in a
  document of its own, which takes what an open keeps (viewport, mode,
  draft resolution, refinement), and replaces the open one only once it
  holds the landscape. A link whose sampling fails (a window whose width
  overflows a double) leaves the open document and its undo history as
  they were, natively and in the page. `restore_landscape` settles; the
  protocol's arm no longer does as well.
- **A link is read within bounds.** The fragment is refused past 16 KB (a
  link is 1.3 KB deflated, 7.5 KB plain); the deflate is streamed and
  stopped past 64 KB out (a link's JSON is 5 to 6 KB); every field the page
  reads is checked for its type before anything is sent. Each refusal is
  said in the notice line, and the page starts at its own landscape; every
  reply of an opening is read, and a refused setting after the landscape
  is said as "opened, but not all of it".
- **Cancels.** Every edit goes through `Engine.edit`, and failures through
  `quietly`, which lets a request the export's Cancel dropped go; a drag
  the engine took hold of is always let go, its end sent again after a
  restart.
- **The formula keeps a draft** typed and not sampled, said as unsampled,
  until return samples it or escape puts the engine's back.
- **The controls' reach is the engine's** (`sliders::reach`): `linkable`
  holds a link to it, and the `catalog` reply sends it to the page, which
  builds its Samples, level and truncation sliders from it. Swift's views
  keep their own copies, which the port does not touch.
- **`camera`** assigns the values it validated.

Tests: four native (an allow-list of kinds; a failed link leaving gamma
and its undo; the catalog's reach; every catalog landscape restoring) and
five in the browser lane (eight malformed links, each refused in 1 to 3 s;
a failed link pasted into a tab; a cancel during a held drag with no error
left unhandled; the formula's draft; the sliders' reach). The lane: 70
passed, 11 skipped by design, 11.8 minutes, run under `caffeinate`: this
laptop sleeps during long runs, and tests that took 7 to 15 minutes in one
run (a long fragment "refused in 408 s") were the machine asleep
(`pmset -g log`: DarkWake), the same test refusing in 0.4 s awake. Export
and the link's round trip are the same bytes as natively in all three
projects. check-web green.

## 2026-11-19 · A pool of Workers: the draft at rest, the judged strokes for Export

The owner's decision on speed (Phase 6, "Interactive timings", options 1 and
2 together): a pool of Workers, and the judged strokes for Export alone.

**The engine's side** (`kurven_session::pool`, `kurven_bake::pieces`). Nothing
in it knows what an engine is or how a share travels, so a desktop host can
run the same shares on threads, each thread a `Session` replaying the same
requests (`kurven-session/tests/pool.rs` does that with sessions in turn).

- **The bake splits at path boundaries.** Swift spreads a bake over its cores
  inside `Heightfield.visibleRuns` (Scene.swift): every vertex of a layer on
  a thread, four at a time, each thread with a traversal forked for it
  (`parallelMap(_:chunk:worker:_:)`), then every fold crossing, then every
  place a run turns, put back in the order the items went in. Each of those
  items belongs to one path, and no path's runs read another's, so a
  landscape layer's runs are its paths' runs put end to end. `plan` cuts each
  carried, clipped landscape layer into pieces of whole paths, about 1,024
  vertices each, never a piece of no vertices (`visible_curves` passes an
  empty set through as it is, where the layer's runs drop vertexless paths).
  A layer derived for the camera (the fold curves, a hatching on the graph)
  is one piece, its stages being serial between them; an unclipped or empty
  layer is one piece; a scene that is not a landscape, or an outline asked
  for, is one piece, the bake itself. `assemble` puts the pieces' strokes
  together in the plan's order. Workers cannot hand each other a vertex at a
  time, so the seam is Swift's, gathered into pieces a Worker is sent whole.
  The forked tests answer as the ones they were forked from, which is what
  lets Swift's split vary with its thread count; here it lets the pieces vary
  with the pool.
- **Requests**, additive in `protocol.rs`: `bakePlan` (the pieces, each with
  its weight in vertices, refused as `bake` refuses and said in the bake's
  status), `bakePiece` (one piece's stroke layers: colours in the JSON;
  widths, points x y z and offsets as f64 and u32 arrays), `bakeAssemble`
  (the pieces given back, answering as `bake` does, SVG and all; or `failed`,
  a piece's refusal said as the status); `restLayers` (the tested layers a
  frame at rest would lay out holding `have`, with weights, and the
  document's fingerprint), `restLayout` (some of them laid out as
  `preview::lay_out` lays them out, with the fingerprint), and `preview`
  given `laid` and `fingerprint` (served under this engine's keys,
  `Document::preview_laying` taking the given layout where `preview` asks
  `lay_out`; refused when the fingerprint is not this document's). Arrays go
  back into an engine with `Session::give` (`kv_give`), for the next request
  alone, and are read back to the bit; nothing goes through JSON numbers (the
  page's `JSON.stringify` writes -0 as 0).
- **The fingerprint** (`Document::layout_fingerprint`) hashes what a frame at
  rest's layouts are a function of: the shown scene's camera, depth tolerance
  and resolution, every layer's curves, and the surface (a landscape's grid,
  window and occluder; a parametric surface's positions; a solid's
  vertices). It does not hash `ContentID`s, which are counted per process:
  two sessions in one process give the same scene different content keys.

**The page's side** (`web/app/src/engine/pool.ts`). The primary is the page's
own engine: it takes every edit and draws every draft. Helpers are engines of
their own, each in a Worker of its own, made when a job first needs them. A
helper holds the document by replaying the primary's journal, as a restarted
Worker does, and then catches up on the primary's **log** (client.ts):
everything the client recorded, in order, each entry marked when it took the
place of the one before (a camera, a slider's move, a setting set outright).
Catching up applies the log since the helper's place, leaving out each entry
a later one took the place of. That is exact by compaction's own claim: the
new entry where the old one was leaves the document both leave. Where the log
no longer reaches back (the journal was cut, `JournalTruncated`) or is longer
than the journal, the helper is given the journal (`adopt`: its Worker
restarted and the journal replayed).

One job runs at a time. Its shares are handed out heaviest first, each to the
helper that comes free first, and put back by the primary in the plan's
order, so the bytes do not depend on the pool's size or on which helper
finished first. A helper whose Worker dies (a trap, a timeout, a cancel) is
restarted by its client, which replays its journal (the job's document), and
its share is handed out again, twice at most. A job asked whether it is
still wanted stops handing out shares once it is not, and lets those in
flight finish. Export's Cancel abandons every job, running or waiting,
restarts the helpers with a share in flight, and leaves the page's engine
alone (before, a cancel restarted the one engine and replayed its whole
document).

Size: a helper for every core but the page's own, at most eight
(`poolSize(navigator.hardwareConcurrency)`). `?pool=N` sets it, and `?pool=0`
is the page's engine alone, as before. Eight because cn gains little past it
(below) and every helper holds a copy of the document.

**At rest.** With WebGPU the draft stays on the canvas at rest. The tested
layers of the frame at rest are laid out by the helpers, a layer each, and
served by the primary from them in a few milliseconds, under its own keys.
The renderer's holdings stay the primary's: a helper's keys start with its
own nonce, and drawing a helper's frames would make the renderer let go of
the primary's resources and take them back at every rest. An epoch, bumped by
every move and change, drops a frame at rest that a move overtook, and a
layout given to a primary whose document moved on is refused by its
fingerprint. A draft waits for neither. Without WebGPU (`fallback`) the
judged strokes at rest are the pool's bake, redrawn at rest, a stale one
dropped the same way.

**The background frame at rest is kept**, for these reasons:

- It is what Swift's window does: drafts while moving, `layOut(draft:
  false)` on another thread at rest, swapped in.
- It costs the hand nothing: it runs on the helpers, and the primary only
  serves it.
- It keeps the draft's known defects off the picture at rest: folds traced
  on the grid, which dash on a steep graph where vertices fall either side of
  the surface (`lay_out`'s comment); wall strokes depth-tested where a wall is
  edge-on; sn's nearly edge-on outline (the draft draws 99.5% of the pixels
  the frame at rest draws, `surfaces.spec.ts`).
- It keeps Phase 7's gate meaning what it says: the preview at rest against
  the judged strokes.

Computing the judged strokes themselves in the background was not done. cn's
take 4.9 s on eight helpers and 34 s on one, for a frame at rest that already
passes the gate at Swift's thresholds.

## 2026-11-19 · Pool timings: the draft at rest, Export at 1, 2, 4 and 8 helpers

Measured in Chrome for Testing 153 (Playwright's) on the Mac's own GPU (the
`gpu` project). This machine has 18 cores (`navigator.hardwareConcurrency`
18), so the default pool is eight helpers. The module is the page's
(opt-level 3). Each pooled bake was run twice and the second taken, the
helpers caught up (`pool.spec.ts`'s measurement, `KURVEN_MEASURE=1`):

| Export (the pooled bake with the SVG) | 1 helper | 2 | 4 | 8 | the page's engine alone |
|---|---|---|---|---|---|
| gamma, the catalog's, at 600 | 2.75 s | 1.45 s | 0.77 s | 0.47 s | 2.74 s |
| cn, the catalog's, at 600 | 33.85 s | 17.58 s | 9.04 s | 4.90 s | 34.27 s |

Every size wrote the same bytes, and the same bytes as the page's engine
alone (gamma 149,334 bytes, cn 813,716). A pool of one takes the page's
engine's time to within 1%: the pieces cost nothing extra, natively either
(cn's pieces take 23.6 s in all against the bake's 23.1 s; building the
traversal once a piece takes a few milliseconds). Eight helpers are 6.9 times
one on cn and 5.9 on gamma. cn's floor is its fold curves, one piece of 1.4 s
natively and about 2 s in a Worker, and the last pieces handed out.

**At rest**, cn at 600, eight helpers, Metal: a camera move's draft is drawn
24 ms after it (engine 16 ms, GPU 3 to 7 ms) and kept at rest; the helpers
start at the rest timer, 0.25 s after the last move; the frame at rest is
swapped in 1.8 s after the move (its fold curves laid out on one helper,
about 1.5 s, then served). On one engine the frame at rest held the engine
for 2.2 s, and a draft asked meanwhile was drawn 2.7 to 2.8 s later on Metal
and 5.5 s on SwiftShader (the entry "What a slow frame at rest forces").
Now that draft is drawn 109 ms later on Metal and 1.3 s later on SwiftShader
(whose draft is 1.4 s of GPU on its own), and `page.spec.ts` holds it under
1.5 s and 4.5 s. preview.spec.ts's frame times: cn's frame at rest 1.54 s
from wish to drawn on Metal (2.2 s before, all of it on the page's engine),
sn's 80 ms. Without WebGPU cn's judged strokes at rest are 4.9 s (35 to 46 s
before).

**Memory**, the module's linear memory per engine after the measurement:
gamma 15 MiB for the page's engine (which also drew every frame), 12 MiB for
the first helper and 8 MiB for each of the rest; cn 34, 27 and 21 MiB. A
Worker's own JavaScript heap comes on top and was not measured. Linear memory
never shrinks, so the page's engine and the first helper, which do the most,
hold the most.

**CI.** ubuntu-latest has 4 cores, so the page there makes 3 helpers. The
browser lane's CI projects (`swiftshader`, `fallback`) run the pool, and the
fallback's frame at rest is the pool's bake. Not measured there.

Under Node (`pool.test.mjs`, in check-web), on the session fixture's gamma
(160 samples) truncated at 3: a pool of 1 in 0.67 s, 2 in 0.38 s, 4 in
0.28 s.

## 2026-11-19 · Swift and the pool

No Swift oddity turned up in this lane. The pool rests on Swift's own claim
for its threads in `parallelMap(_:chunk:worker:_:)`: each thread's forked
worker answers as the one it was forked from, chunks are handed out as
threads come free, and results are put back in order. It held here: every
split tried gives the bake's strokes to the bit (`kurven-session/tests/
pool.rs`: gamma and the cubic cut into pieces of 1, 7, 300 and 2,048
vertices and whole, baked last piece first; pools of 1, 2, 3 and 5 sessions;
the browser lane's pools of 1, 2, 4 and 8 on cn and gamma).

One thing Swift does that the pool does not: Swift's window lays out its
frame at rest with every core, the folds included (the fold placer's stages
in parallel). The pool lays out a layer per helper, so cn's fold curves are
one helper's 1.5 s. Splitting the fold placer's stages across Workers would
take a round trip per stage (rows, vertices, crossings, chords, ends); it was
not done.

## 2026-11-19 · The pool's gate, and what was run

- **check-web** gains "the pool of engines under Node" (`web/app/test/
  pool.test.mjs`, against the check's module and the page's): pools of 1, 2
  and 4 bake the bytes the page's engine bakes alone and the bytes the native
  session wrote (gamma-truncation), every stroke array too; helpers catch up
  on sixty gestures and a drag's moves; a helper whose Worker is terminated
  while its piece is posted is restarted and the piece handed out again; a
  cancel ends a bake with `Cancelled` and the next bakes; a bake no longer
  wanted is stale; a frame at rest served from the helpers' layouts is the
  primary's own, and refused once the primary's camera moves. Natively,
  `kurven-session/tests/pool.rs` (five tests, above). The Worker protocol
  test is unchanged and passes with the client's additions (inputs, the log,
  `snapshot`, `adopt`).
- **The browser lane** gains `pool.spec.ts`: Export of a typed expression
  through pools of 2, 1 and 4 is native's bytes (`kurven-cli session` on the
  page's journal) and the page's engine's alone, in every project; at rest
  the draft stays and the helpers' frame at rest is swapped in (WebGPU
  projects). `page.spec.ts`'s slow frame at rest now asks that a draft not
  wait for it, and holds the engine with a frame at rest of its own to keep
  testing that gestures given meanwhile go as one request. The whole lane:
  75 passed, 15 skipped by design, 10.3 minutes (11.8 before), run under
  `caffeinate`. Every coverage figure is the one before: the frame at rest
  is the same strokes, laid out elsewhere.
- **Not run**: the lane on Linux or in CI; the measurement test there.

## 2026-11-19 · Review of PR #76: shares name their document, drifted helpers come back

A code review gave ten findings. Each was checked against the code; all ten
were real, and all ten are fixed. None was rejected. Where a fix stops short
of what was asked, it says so below.

1. **A helper that drifted was never brought back.** It is now. Every share
   names the fingerprint of the document it was asked of: `bakePiece` and
   `bakeAssemble` the bake's (`Document::bake_fingerprint`: the framed scene,
   the depth tolerance, the bake's resolution), and `restLayout` the frame at
   rest's. An engine holding another document refuses the share with
   `another: true`. The pool then marks the helper (`dirty`), restarts it and
   replays the primary's journal into it (`adopt`), and hands the share out
   again, twice at most. A helper is marked the same way when a share fails
   other than by a cancel or a dead Worker (a replay cut short,
   `JournalTruncated`, among them), and when it refuses a layout. Tested in
   `pool.test.mjs`: every helper's document changed behind the pool's back,
   then one helper's, then a frame at rest with every helper drifted; each
   time the bytes or the frame are the primary's. `JournalTruncated` itself
   is not forced in a test: it takes the same path as any failure that is not
   a cancel.
2. **The log grew without bound.** A client now keeps the log only once a
   pool asks it to (`keepLog`), so without a pool (`?pool=0`) and on a
   helper's own client nothing is logged. It is trimmed with the journal to
   twice the journal's length plus sixteen: a helper further behind than that
   replays the journal rather than catch up, so nothing older is of use.
   Tested: 120 viewports and a 40-move drag leave the log within the bound,
   the helpers' logs empty, and a client with no pool logs nothing.
3. **An assembly after the document moved on mixed two documents.** The
   pieces are all of the plan's document, since a piece of another is
   refused. The decision is to **export the plan's document consistently**,
   as one engine's bake, asked before an edit, baked the document as it was.
   `bakeAssemble` carries the plan's fingerprint and frame. When the primary
   still holds that document the reply is `bake`'s. When it has moved on,
   the strokes and SVG are the plan's document's, the frame is the plan's,
   the reply says `snapshot`, and the bake's status says "of the document as
   it was when the bake was asked". The page downloads it, and its status
   line says so. Tested natively: a move and a truncation between the pieces
   and the assembly give the bytes and frame of the bake before them.
4. **A cancel during assembly still downloaded the SVG.** The bake now
   rejects with `Cancelled` when the job was abandoned while the primary put
   the pieces together, and the page writes nothing. Tested.
5. **Every refusal of a given layout was swallowed.** Now only a frame that a
   move overtook (the epoch moved) is let go quietly. Any other refusal is
   logged, counted (`view.fallbacks`), and the frame at rest is laid out by
   the page's engine itself, so the rest still refines.
6. **A refused piece reported the first refusal in time.** It now reports the
   first in plan order. Once a piece at index k is refused, no piece after k
   is handed out, those before it still are, and the lowest refused index is
   reported. Tested with two pieces refused on purpose, the later one
   answering first: the bake reports the earlier, as its status too. No real
   piece can be refused today: the port constructs no `PortRefusal`, and a
   piece of another scene is refused as of another document. So the
   equivalence with the single engine is tested on the refusals that do
   happen, before the pieces: a perspective camera gives the same error and
   the same status pooled and alone (natively and under Node).
7. **`rest_layers` ran a whole preview.** It is now keys alone: for each shown
   layer tested for the camera, its key at rest (`preview::rest_key`, which
   `preview` itself now uses for the same keys) against what the page holds.
   It builds no resources, lays out nothing, and leaves the preview's memo
   alone. Tested: holding the frame at rest's keys it wants none, holding
   none it wants exactly the tested layers.
8. **The fingerprint was rehashed on every ask.** It is now cached per slot
   (the frame at rest's, the bake's), keyed on the scene's content and
   curves, its camera, depth tolerance and resolution, and the bake's
   resolution. A `ContentID` is counted per process and is new whenever a
   scene's geometry or curves are (the preview's keys rest on the same
   claim), so the key changes whenever what is hashed can.
9. **Two copies of the bake's preconditions, and of `have`'s parsing.**
   `Document::bake` now asks the same `bake_inputs_said` the plan does, and
   both `preview` and `restLayers` read what the page holds with one parser
   (`holdings`).
10. **A pool of at least one helper on a one-core machine.** `poolSize` is 0
    for one core or none reported, so the page runs on its own engine.

Both gates were green after the fixes; the counts are in the report of this
round.

## 2026-11-19 · Second review of PR #76: the whole document fingerprinted, the primary as fallback

A second code review gave ten more findings. Each was checked against the
code; all ten were real, and all ten are fixed. None was rejected.

1. **The fingerprint left inputs out.** Each layer's source (a hatching's
   spacing, angle, tone and faces), the curves' surface coordinates, and the
   heightfield's refiner, walls, tiles, region and step were not hashed. A
   helper one hatching-slider move behind would have passed the check. Now
   `content_hash` hashes every input the bake and the layouts read: each
   layer's spec as a manifest writes it (`LayerSpec::json`), its curves and
   coordinates; a landscape's grids of |f| and arg f, truncation, cache flag,
   occluder (vertices and triangles), tiles, region, step, walls and
   refiner; a parametric surface's grid, axes, outward side, neighbourhood
   and jet; a solid whole (its derived `Debug`, every field).

   A refiner is closures, so no value of the scene names its function. The
   open bundle's manifest, from which the refiner is built (`Document::read`,
   `refiner_for_bundle`), is hashed for it, with whether one is attached.
   The finding asked whether the fingerprint could simply be derived from
   the scene's `Debug`; it cannot. `ContourRefine`'s, `ParametricSurface`'s
   and `Mesh`'s `Debug` print a summary (a tolerance, the axes, the counts),
   and the scene's per-process `ContentID`s and fold memo would differ
   between engines holding the same document.

   **The cache key.** The costly part is cached on the scene's content and
   curves and the manifest's hash; the camera, depth tolerance, resolution,
   mode and the bake's options are hashed on every ask, so a camera move
   rehashes nothing. A scene's fields are private and only its constructors
   make one, each minting new IDs: `heightfield_scene` and the others mint
   both, `drawing` keeps the content (the geometry is the same) and mints the
   curves, `looking` keeps both and changes only the camera. So the key
   changes with every input hashed.

   **Tests.** `pool::tests::every_input_of_the_bake_changes_the_fingerprint`
   changes each input alone on gamma (a hatching's spacing and angle with
   its paths kept, a clip, coordinates, the refiner, the walls, a tile more,
   a tile moved, the region, the step, the cache flag, the manifest) and
   requires a new hash. The same scene rebuilt, with new IDs, keeps it.
   `a_piece_asked_before_any_edit_is_refused_after_it` asks a piece of the
   document before each edit and requires it refused after: a cap
   hatching's strokes, a wall hatching's, a level count, the refinement, the
   truncation, the depth tolerance, the bake's resolution.
2. **A failing pool failed the frame at rest and Export.** Now a job that
   fails for the pool's own reasons is said on the console and done by the
   primary: the view lays out the frame at rest (or bakes, without WebGPU)
   on the page's engine, and Export bakes there, the same bytes. After three
   failures in a row the pool is switched off for the session
   (`EnginePool.disabled`), its helpers closed, said on the console, and the
   page runs as with `?pool=0`. Tested: three bakes whose helpers will not
   bake switch the pool off, said so, its helpers closed.
3. **A refused layout left the draft up.** A layout a helper refuses now
   marks it, brings it back from the primary's journal, and asks it again
   within the same frame at rest. Past its tries the job fails, and the
   primary lays out the frame (point 2). Tested with a helper that refuses
   its first layout.
4. **Export's cancel abandoned every job.** Jobs now say what they are for
   (`export` or `rest`); Cancel abandons the export's alone and restarts
   only the helpers holding its shares. Tested: a frame at rest waiting
   behind a cancelled export is laid out.
5. **A resize closed helpers in the middle of their shares.** A closed
   helper now gives back the share it holds (no try counted) and takes no
   more; the helpers left take them. Tested: a pool of four resized to one
   in the middle of a bake gives the same bytes.
6. **Export waited behind an unwanted frame at rest.** An export asked for
   while a frame at rest that is no longer wanted runs or waits now abandons
   it and restarts the helpers holding its shares, rather than wait for them
   (cn's fold curves, about 2 s). Tested with layouts held back a second: the
   export is the same bytes, 0.7 s later, and the frame at rest ends without
   drawing.
7. **A one-piece plan used the pool for nothing.** A surface, a solid, or a
   bake with an outline asked for has nothing to share. `bakePlan` with
   `bakeAlone` now bakes a plan of one piece on the primary and answers as
   `bake` does, with `baked`. Tested natively on sn and the torus, and under
   Node on the torus bundle, the same bytes. The first frame at rest on cn
   does not wait for eight helpers: a job makes only as many helpers as it
   has shares (cn's frame at rest three: the walls' two layers and the fold
   curves), and each starts its share once its own replay is done, with no
   wait for the others.
8. **Inputs were copied before transfer.** A request may now transfer its
   inputs (`Options.transfer`): each array that is the whole of its buffer is
   transferred, not copied. The pool does so for the pieces it assembles and
   the view for the layouts it serves, neither of which is read again; other
   callers keep the copy.
9. **The two kinds of share were weighed by different rules.** There is now
   one rule (`pool::weight`): a range of paths weighs its vertices; a whole
   layer its vertices; a layer derived for the camera (the fold curves, a
   hatching on the graph) at least what the largest layer does; an unclipped
   layer nothing. Before, a hatching on the graph was weighed by its carried
   vertices at rest (none) and every whole layer as the largest in the bake.
10. **A plan that left paths out was put together.** `Piece::Paths` now
    carries its layer's path count (`of`); a helper refuses a piece whose
    count is not its layer's. `pieces::covers` refuses a plan that is not
    each layer in turn from the first, whole or as pieces running from its
    first path to its `of`th with no gap or overlap, or that has a whole bake
    with anything else; and, where the assembling engine still holds the
    plan's document, one that draws fewer layers than the scene. `assemble`
    refuses an uncovered plan. Tested on eight bad plans (the last piece
    short, a gap, an overlap, a layer not from its first path, out of turn,
    twice, a whole bake with more, counts that disagree) and a layer left
    out.

## 2026-11-19 · Phase 8, step 4: the picker

`kurven_session::thumbnails` is Thumbnails.swift's engine half: each catalog
entry (`Entry`, Gallery.swift's `GalleryEntry`: the functions, then the
surfaces) built as Swift builds it -- a function sampled at 220 along its
longer side and hatched at a twenty-second of its window, fitted to its exact
bounds (`view_bounds`, falling back to the quick ones); a surface on a grid of
256 round its ring, the forced trajectory a quarter as long, fitted to its
quick bounds -- the camera it opens at fitted to 440 by 300. The protocol's
`thumbnail` request answers the frame the window's preview draws of it at
rest, every layer shown, in the drawing mode, or with `judged` its judged
strokes, for a page without WebGPU. Nothing of the open document is touched:
`Document::preview` is split so `preview::preview_scene` draws any scene from
a navigator, viewport and mode, with a memo of its own. The `catalog` reply
lists the entries, each with the key its picture is cached under: the id and
FNV-1a of what would change the picture, as Swift's `cache(for:)` hashes it.

The page's picker (`page/picker.ts`) is Gallery.swift: a dialog, a filter
over name, label, formula and notes, the two groups, each cell its picture,
label and formula, the notes as its title; Browse... beside the dropdown and
in the Surface section opens it. The pictures come from a second engine, a
Worker of their own started the first time the picker opens, one at a time
in catalog order, so the document's engine is never behind them (Swift draws
them one at a time on a detached task for the same reason). Each frame is
drawn by a second `PreviewRenderer` on the window's device into a 440 by 300
texture (or, without WebGPU, the judged strokes on a 2D canvas), read back,
and cached as a PNG in IndexedDB under the engine's key and the kind of
picture; shown at half size, as Swift shows its own.

Native times, one thread (`tests/thumbnails.rs`, ignored): preview 28 ms
(sqrt) to 695 ms (cn) and 1,038 ms (the forced torus, its fit), 4.1 s for
all seventeen; judged strokes 77 ms to 4.8 s (cn). In Chromium on Metal the
picker test draws all seventeen and checks them in 13 s; a second page reads
all seventeen from the cache and draws none.

Differences from Swift, by construction: Swift's cache key spells numbers as
Swift prints them and the port as Rust's `{:?}` does, and its style counter
starts again at 1 (another renderer draws the pictures): the caches are not
shared and need not be. The judged pictures are baked at the window's
quarter pixel; the preview pictures, as Swift's, from a scene whose
resolution is Swift's default.

## 2026-11-19 · Phase 8, step 5: surfaces and their controls

`kurven_session::surface_controls` is SurfaceSection.swift's model: each
field's label, reach, step, whether it slides (a refit's two do not), its
digits and help (`control`); where a slider rests (`detents`: the slope at
fractions with denominators up to eight, or on a forced torus at its own
rotation number Ω/ω and its convergents; the trajectory's turns at the
convergents' denominators), `quantize` as Swift's (the first nearest detent
within reach, else the step); the linear curve described in words; and the
edits the section's switches and pickers make, each one undo step named for
the control: the coordinate curves on at 36 by 18 ("Coordinate Curves"),
their counts clamped to 1...200 ("Lines Around", "Lines Across"), the
linear curve on at the trajectory's own slope or 2/5 for 24 turns ("Linear
curve"), a forced torus's components swapped when the other's is chosen
("Radius from", "Height from"). The state carries it (`surfaceControls`);
the protocol takes `surfaceField` (a number typed, stepped or switched,
clamped to the reach as Swift's binding clamps it), `surfaceLines`,
`surfaceLineCount`, `linearCurve`, `component`, and a drag's move with
`rest`, which the engine rests at the step or a detent.

The page's Surface section (`page/surface.ts`) builds its rows from the
state: per number a slider (min, max and step the engine's) and a field
whose arrows are the stepper, or a field alone; the components as segmented
x, x′, x″; Conformal; the coordinate curves' switch and counts; the linear
curve and its description; Browse...; the status. torus, sn and the forced
torus come from the picker (and `?surface=`); `periodic` from painting a
function (step 6) or a bundle; sn in relief from a bundle (step 7), as in
Swift, whose catalog has no relief.

Tests: `tests/surface_controls.rs` (the rows and reach, a value clamped, the
slope rested at 2/3, 3/5 and the step, quantize's ties, sn's switches, the
forced torus's components, convergents and own slope); `surface.spec.ts`
(a torus dragged, typed past its reach, counted, its linear curve on and its
slope dragged, undone; sn's curves, modulus dragged, conformal; a periodic
function's samples; sn in relief opened, its relief typed, clamped, undone),
each document's Export held to `kurven-cli session` on the page's journal,
byte for byte. `kurven-cli session --stage NAME DIR` stages a bundle first.

## 2026-11-19 · Phase 8, step 6: bands, picking and painting, undo, polyhedra

- **Banded truncation** (TruncationSection.swift's band rows): "Per band of
  Re" starts one band at the window's middle and a beyond at the current
  height, as Swift's `apply(.bands)`; each band's threshold, its height
  slider (to max(3 h, 10), from the engine's least) and a way to take it
  off; Beyond; Add a band. A height dragged is one undo step,
  "Truncation": the protocol's truncation drag takes the whole truncation
  as a move (`truncation`), as Swift's slider writes `.realBands(next, ...)`.
- **Picking**: a click held still on the drawing sends `pick` at its pixel,
  after the gestures before it (GestureView's mouseUp; a click that orbits,
  or a second click, picks nothing). The engine casts the pixel as Phase 6
  made it do.
- **Curves on** (PaintSection.swift): `paint_caption` ported (Swift's
  `paintCaption`); the layers on the face with their paths, own controls and
  a way to take each off; the Add menu (offered once the drawing is the one
  asked for); Function... where `offers_function`, a sheet with the
  expression and its periods: the `periods` request, a view, gives the
  catalog's rectangle as the expression is typed (`known`) and the one the
  sheet opens with (`offered`), as Swift's sheet; Paint refused stays in the
  sheet and says why.
- **A layer's controls** (LayerControls), shared by the Layers list and
  Curves on: levels, strokes (2...400), the plateau hatching's axis, a
  light's azimuth and elevation (the engine sends the angles,
  `LightSlider.angles(of:)`). The Layers list shows each layer's paths and
  "unclipped", as Inspector.swift's `layerList`.
- **Undo and Redo** as buttons named for the step, beside ⌘Z and ⇧⌘Z.
- **Polyhedra**: the Polyhedron section (its solid, boundary, genus or
  shells, from the engine) and a hatching's strokes; opened from a file
  (step 7), as in Swift.

Tests: `tests/paint_page.rs` (the caption and Function... on a torus, the
periods, a painted function offered back, a band's height dragged as one
step and undone whole, a polyhedron's facts, paths and an axis in the
state); `paint.spec.ts` (bands by their rows; the surface picked by a click,
an orbit that picks nothing, sn's periods coming with its name, a bad
function refused in the sheet, a curve added, its levels stepped, taken off,
undone and redone by the buttons; the picture frame's strokes dragged and a
face of it hatched), every Export the native bytes.

## 2026-11-19 · Phase 8, step 7: a .kurven bundle as a file

A bundle is a directory, which a browser can neither download nor be handed
as one file. **Decision**: Save bundle writes its files into a ZIP archive
under the bundle's directory name (`gamma.kurven.zip` holding
`gamma.kurven/manifest.json`, `height.npy`, `phase.npy`: unzipped, the
bundle Swift's app and kurven-cli open), stored, dated 1980-01-01, so one
bundle always writes the same bytes; the directory's name is the catalog
entry's name, or the title, in ASCII (macOS's `unzip` cannot write a
non-ASCII name). Open takes such an archive (stored or deflated), a bare
manifest.json (a surface's or a polyhedron's whole bundle), or a bundle's
folder (`webkitdirectory`). A desktop wrapper can open and save directories
with the same requests: the archive is the page's (`page/bundle.ts`), the
bounds and the atomic open the engine's.

**A file is untrusted input.** The page reads it within bounds before the
engine sees it: the archive at most 160 MB, at most 64 entries, each at most
64 MB and 128 MB in all unpacked (a 2000-sample landscape's two grids are
64 MB), a deflated entry inflated as a stream and stopped past its cap, each
CRC checked; no encryption, no ZIP64, no path that climbs out, a single top
directory taken as the bundle's. The engine then holds the manifest to the
controls' reach before anything is built (`kurven_session::files::openable`,
parsing the manifest of its kind alone): a landscape's truncation finite and
at most 64 bands; a surface's request past Swift's `check` within the
controls (the function's samples 120 to 1200, the fit, the harmonics, the
trajectory's turns, the coordinate curves 1 to 200, the linear curve's turns
and strands) and its ring at most 16,384 samples (twice what `kurven-cli
surface sn --relief` builds: the relief fixture is 8192; Swift allows
65,536); at most 64 layers, 1000 levels a contour, 400 strokes of a hatching
or 400 seeds of slope lines across the drawing's extent (a landscape's
window, a solid's box, a surface's rectangle), coordinate and linear curves
within their controls, every number finite. The protocol's `open` with
`untrusted` opens it in a document of its own, which takes what an open
keeps (viewport, mode, draft resolution, refinement), put in place only once
open: a refusal or a bundle that will not read touches nothing (`touched`
false, so the client journals no open), and the document and its undo
history stay. An opening that runs past two minutes -- a function that never
returns, `besseli(1e12, z)` painted on a torus -- is cancelled by the
client: the engine is restarted and the journal replayed, so the document is
the one before. The bytes of a refused bundle stay staged (the client
journals a stage when the engine takes it), which costs memory until the
next restart and nothing else.

Tests: `tests/files.rs` (all eight session inputs open from a file; nine
refusals -- no manifest, not JSON, the wrong schema, 2000 levels, a hatching
too fine, a ring of 60,000, sn's samples at 3000, coordinate curves at 5000,
a hundred bands -- each touching nothing, the undo after them undoing the
edit before); `files.spec.ts` (gamma's folder opened and edited, its Export
the native bytes; Save bundle's archive holding exactly the files
`kurven-cli session --save` writes, byte for byte, and opened in a fresh
page to the same drawing; sn in relief and the cube from their manifests;
noise, a garbage manifest, an archive without a manifest, one that climbs
out, 2000 levels, a ring of 60,000, each refused and said with the document
unchanged; a function that never returns cancelled after the page's
`openTimeout`, the engine restarted once, the document and its undo as
before).

## 2026-11-19 · Steps 4 to 7: the engine's answers, natively and in wasm

`crates/fixtures/page/*.jsonl` are the requests the page sends in steps 4
to 7: the catalog and six thumbnails (frames at rest and judged strokes, a
function and each kind of surface), a surface's controls, a band dragged, a
face picked and painted, a polyhedron's strokes and a face hatched, bundles
opened from files (one refused) and saved. check-web runs each through
`kurven-cli session --replies` and through the page's module under Node
(`kurven-wasm/requests.mjs`) and requires every reply, JSON and arrays, to
be the same bytes: 198 files, identical. The browser lane holds every new
kind of drawing the page can now make -- a torus with its lines and linear
curve, sn conformal, a periodic function, sn in relief, the picture frame,
the cube, a banded truncation, a painted torus, a landscape opened from a
folder and from a saved archive -- to `kurven-cli session` on the page's
journal, byte for byte, in every project.

## 2026-11-19 · Where the Swift looks wrong (steps 4 to 7, ported as it is)

- **A surface value refused is said nowhere visible.** `SurfaceFieldSlider
  .set` writes the refusal to `paintStatus`, which only the Curves on
  section shows, and only while a face is picked; with nothing picked a
  refused relief or radius just does not move. The page also says it in the
  Surface section's status line (a view's detail).
- **The fold curves' count is 0.** The Layers list shows each layer's
  `paths.count`; a fold layer carries no paths (its curves are traced for
  each camera), so it always reads 0, in Swift and here.
- **Two controls named "Turns".** On a forced torus with a linear curve, the
  trajectory's turns and the linear curve's turns are both labelled
  "Turns", and so are their undo steps.

## 2026-11-19 · Phase 8 status (steps 4 to 7)

**Done**: the picker, surfaces and their controls, banded truncation,
picking with Curves on, painting a function, undo and redo, polyhedra and
their hatching, and `.kurven` bundles opened and saved as files, each with
its engine half natively tested, its requests byte-identical in wasm, and a
Playwright spec (`picker`, `surface`, `paint`, `files`) whose Exports are
the native bytes. **Protocol additions**: `thumbnail`, the `catalog` reply's
`entries`; `surfaceField`, `surfaceLines`, `surfaceLineCount`,
`linearCurve`, `component`, a drag's `rest` and a truncation drag's
`truncation`; `periods`; `open` with `untrusted`; in the state,
`surfaceControls`, `polyhedron`, the pick's `caption`, `takesFunction` and
`expression`, each layer's `paths`, `clipped`, `lightAngles` and `axis`.
`kurven-cli session` gained `--stage`, `--save`, `--replies` and comments.

**The gate**: check-web green (its new step: 198 reply files of the
page's requests byte-identical, native and the page's wasm; the goldens,
the session's 28 scripts and 87 files unchanged). The browser lane
(`KURVEN_PORT=4191 scripts/check-browser.sh`: another agent's lane held the
default port 4175, so the port is an environment variable worth knowing):
100 passed, 11 skipped by design, 20.2 minutes under `caffeinate`, the four
new specs passing in swiftshader (offscreen), gpu (Metal) and fallback. The
picker drew all seventeen pictures in 14 s on Metal, 24 s on SwiftShader and
66 s without WebGPU (judged strokes, cn's 4.8 s natively the longest). The
phone-width test now tabs through Browse..., which sits under the
dropdown's notes as in LandscapeSection.swift; Swift's File and Edit menus
are a section after Export.

**Not done**: a link to a surface or a polyhedron (both still write none;
`linkable` is unchanged, so a link stays a landscape's); double-click to
retarget (Swift's mouseDown with two clicks: the engine would need the
world point under a pixel as a request); the Camera section's Perspective
switch and azimuth and elevation fields, the Visibility section's depth
tolerance field, the Bundle section's provenance, Swift's literal sliders.
The new specs have run on this Mac only, not on Linux or in CI.

## 2026-11-19 · Review of PR #77: untrusted files, staging, the light, clicks

Ten findings of a /code-review pass, each checked against the code:

1. **A light with no direction.** Fixed where it was real. A manifest's and
   a link's isophote are read by the schema, which already refuses a light
   with no direction (`Tone::has_direction`: squared length positive and
   finite), so a file or a link with a zero light was refused all along
   (now tested: the file is refused and the document unchanged). The
   `light` request was not: `[0, 0, 0]` was set on the layer, and the
   state's `lightAngles` normalized it to NaN, which the canonical writer
   asserts on, a trap in wasm. The request now refuses a light without a
   direction, touching nothing (zero, one whose length underflows, one
   whose length overflows); `light_angles` is total (no angles for such a
   light, and none that are not finite); `files::usable_light` is
   `Tone::has_direction`, and the link allow-list uses it too. Swift's
   `simd_normalize` of a zero light is NaN, and its sliders would show NaN.
2. **An unordered window.** Fixed. `openable` measured spacings against
   max(real.length, imag.length) of a domain the schema reads without
   ordering it; a window read lo > hi gave a negative extent, which every
   spacing bound passes. A landscape file's window must now be lo < hi,
   finite, each way; a surface's rectangle is measured by the absolute
   value of its sides (a periodic function's periods may be negative) and
   must be finite; `layers` refuses an extent that is not a length. The
   link allow-list already refuses a window that is not lo < hi, finite,
   so its bounds had no sign hole; the polyhedron's extent is a box's,
   never negative.
3. **Wall and riser hatching against an unbounded height.** Rejected. A
   wall's or a riser's hatching is vertical strokes (its angle is held to
   π/2) spaced *along* the wall or the riser: `wall_hatch_lines` and
   `riser_hatch_lines` take `stroke_count(length, spacing)` of the edge's
   or the riser's horizontal length, and each stroke is its two ends, so
   the truncation's height moves where a stroke ends and not how many
   there are. The window's extent bounds the count, which `openable` and
   the link allow-list already hold to 400 across. Held by a test: gamma's
   wall hatching opened at a truncation of 5 and of 1e7 draws the same
   number of paths. A truncation height has no upper reach in Swift (its
   slider tops out at max(3 h, 10) and its field takes any height), so
   there is no control's reach to hold one to.
4. **Staging under a name merges.** Fixed. The engine adds a staged
   bundle's files to whatever is staged under its name, so a second
   gamma.kurven.zip lacking phase.npy read the first's. Each file is now
   staged under a name of its own (`opened/N/gamma.kurven`; the title and
   the refusals read the last part), and the engine lets go of it after
   the open (5). Tested: a second archive of the same file name without
   phase.npy is refused, the document unchanged.
5. **Opened bytes kept until the page closes.** Fixed. After every open
   from a file, opened or refused, the page sends `unstage` (a new request:
   the engine drops the staged bundle; the document holds what it read).
   **The approach for the journal**: it keeps the raw staged files of the
   bundle the document is, and of no other. A replay needs them to open
   that document again, and a canonical bundle saved from it would be as
   large (a landscape's two grids), so the compact form is the one bundle,
   not every bundle ever opened. In the client (`engine/client.ts`,
   `record`): an `unstage` drops the stages of its name unless a journaled
   `open` reads them, in which case it is kept after them, so a replay lets
   go of the bytes again; an opening that wipes the slate drops the stages
   of bundles unstaged, but for the one it reads. A bundle never unstaged
   stays journaled as it stays staged (the engine keeps it, and a later
   open may read it: the Worker test's opens of gamma after the torus).
   The browser lane checks that the journal holds one bundle's bytes after
   each open and none after a refusal.
6. **paint_caption cloned the scene.** Fixed: `faces_now` borrows the
   document's scene (the caption and the Add menu both use it now).
7. **The double click and the drag.** Fixed in part. A pointer event's
   `detail` is always 0, so a double click picked twice; the page now picks
   on the `click` event, the first of its count (a test counts one pick
   for a double click). The threshold is Swift's: any move of the mouse
   with its button down is a drag (AppKit's `mouseDragged`; GestureView has
   no threshold), so a mouse or a pen keeps that; a finger, which Swift has
   none of, may stray 6 CSS pixels and still pick.
8. **A detent outside the range.** Kept, as Swift: the slider's `set` takes
   what `quantize` returns, unclamped. No detent of the catalog's surfaces
   lies outside its range (fractions up to 2; the forced torus's Ω/ω ≈ 0.42
   and its convergents, denominators up to 600), and a value the request
   refuses is refused there. Recorded below as a Swift oddity.
9. **The thumbnails' Worker never ended.** Fixed: it is closed once every
   picture is drawn, cached or failed, and when the picker closes (after
   the picture being drawn); a later opening starts one again. Tested: the
   engine is not running after the pictures are drawn, nor on a page that
   read them all from the cache.
10. **Duplicated checks.** Fixed: `files::line_counts` is the one check of
    the coordinate curves' counts, and `slid` clamps through `typed`.

Where the Swift looks wrong: **a slider's detent is not clamped to its
range** (SurfaceSection's `quantize` result goes to `set` as it is), so a
forced torus whose rotation number lay past 2 would rest its Slope slider
outside the slider's own range; ported as it is.

**CI's first run of PR #77** failed at the page's requests step on Linux:
`crates/fixtures/page/levels.kurven/` was left out by .gitignore's
`*.kurven/` and never committed, and the step discarded the native
session's errors, so it said nothing of why. The bundle is negated in
.gitignore as the session's inputs are and committed; the step's errors
are printed; and the step first requires every file it reads
(`crates/fixtures/page`, the session's inputs) to be the commit's -- none
ignored, none untracked, none changed since (`committed` in check-web.sh,
checked failing for an ignored, an untracked and a staged fixture) -- so a
missing fixture fails on this Mac first.

## 2026-11-19 · Steps 4 to 7 with the pool (main merged, PR #76)

`origin/main` (b7e42ec, the Worker pool) is merged into this branch.

- **The journal and the pool's log.** The pool's helpers catch up on the
  primary's log or start again from its journal (`snapshot`). `compact`
  keeps the unstage rules (above, "Review of PR #77", 5) and logs every
  `unstage`, whether or not the journal keeps it: a helper that has staged a
  bundle from the log lets go of it as the primary did, and one that has not
  lets go of nothing. A helper that starts from a snapshot replays the
  journal, which holds the stage, the open and the unstage of the bundle
  the document is, and of no other bundle opened from a file.
- **`preview_scene`** takes the pool's `lay` (a frame at rest whose tested
  layers the helpers lay out), so `Document::preview_laying` is
  `preview_scene` with the document's scene, and the thumbnails pass
  `lay_out`. The pool's `bake_reply` and the thumbnails' judged strokes
  share `protocol::stroke_arrays`. Thumbnails keep their own client,
  outside the pool.
- **Pooled, as alone, as native.** `tests/pool.rs` bakes seven of the new
  scenes through a pool of sessions to the bytes the session bakes alone: a
  torus with its radius, lines and linear curve; sn conformal with its
  coordinate curves; the forced torus with its components swapped; a torus
  painted with sn and a curve added; sn in relief, the picture frame (its
  strokes and a face hatched) and gamma (banded, its band dragged, its graph
  painted) each opened from a file under a name of its own and let go of. A
  surface or a solid is one piece and the primary bakes it; the banded,
  painted gamma is shared. `pool.test.mjs` runs the same scenes through the
  page's client and an `EnginePool` of four helpers in Worker threads
  (pooled as alone; the page's engine is native's by check-web's requests
  step), and checks that the journal then holds one bundle from a file.
  `pool.spec.ts` exports gamma opened from its folder, banded and painted,
  the frame from its manifest and a painted torus through pools of 2 and 1,
  each the bytes `kurven-cli session` writes from the journal, the opened
  bundle staged natively under the page's name. Every other spec of steps 4
  to 7 runs with the page's default pool, so their Exports are pooled too.
- **The fingerprint.** `tests/pool.rs` holds that a band dragged, the
  plateau hatching's axis turned and a curve painted on gamma's graph each
  make a piece asked before them refused after, and that a torus's radius,
  its lines' count, its linear curve and a function painted on it each make
  a layout at rest asked before them refused after.

## 2026-11-19 · Second review of PR #77: one set of bounds, an open as by name

Ten findings, each checked against the code:

1. **A hatching's `step`.** Rejected, with tests. The step is read only on
   a solid's faces (a line's tone, every `step` along it), and Swift's
   `PolyhedronManifest` already bounds those readings to 2^20 in all ("the
   hatchings would read their tones 2.42e+06 times, more than the 1048576
   a polyhedron's hatchings read": the frame's hatching at a step of 0.0008
   is refused, tested). A landscape's hatching, which is all a link holds,
   may carry no step at all: its schema refuses one ("a line is its two
   ends on a wall or a riser ... so schema 3 writes none"; a link with its
   walls' step at a millionth of the spacing is refused, touching nothing,
   tested); and a surface's hatching runs along its grid, not read every
   step. So no step reaches a reading the schema does not bound.
2. **The picker reopened while stopping.** Fixed. `warm` handed back the
   run that was stopping, and nothing warmed again. Now the pictures are
   *wanted* while the picker is open: a run stops after the picture under
   way once they are not, `warm` waits for a run under way and starts
   another for what is left, and the engine is let go of only when no
   picture is wanted or none is left. Tested by closing and opening again
   mid-draw (every cell drawn, none spinning, the engine let go of after);
   on the old code the same test waits forever.
3. **An untrusted open built its own document.** Fixed, more simply than a
   successor: `load_from` is now `read` then `load_read`, and an untrusted
   open checks `openable`, reads the bundle whole (`Document::read`,
   refused touching nothing if it does not read) and then opens it with
   `load_read`, exactly as an open by name does. Tested: gamma, sn and the
   frame opened from a file after a session set the depth tolerance, the
   bake's resolution, the mode, the refinement, the draft resolution, the
   viewport, a pick's status and an edit to undo give the very state the
   same bundle opened by name gives. (Either way the depth tolerance and the
   bake's resolution are then the bundle's first preset's, as Swift's
   adoption sets them; neither path keeps the session's.) `openable` now
   also refuses a landscape without a camera preset, the one failure the
   adoption itself finds. A link's landscape is still made apart, since its
   sampling can fail, in `Document::successor`, which keeps what `create`
   keeps (the viewport, the mode, the draft resolution, the refinement,
   the depth tolerance, the bake's resolution and the status lines but the
   landscape's): tested against `create` after the same settings.
4. **A restriction's bands.** Fixed, and more than the finding. A
   restriction's terms are checked (finite, at most 16 with intersections
   unfolded), in files and links. Checking it turned up the wider hole: the
   JSON reader takes a number past a double's range (`1e400`) as infinite,
   and an infinite camera angle, tile matrix entry, wall corner or depth
   tolerance in a manifest was written back into the state, where the
   canonical writer refuses it -- a trap in the wasm module. Every number of
   every fixture manifest replaced in turn by `1e400` trapped the engine at
   60 places (the presets and the occluder). A manifest, and a link, holding
   any number past a double's range is now refused whole before a field is
   read (`files::every_number_finite`); the same replacement by `1e400`,
   `1e300`, `-1e300` and `0` traps none of them. A window whose width
   overflows (-1e308 to 1e308) is now refused as no interval, before its
   sampling fails; the tests of a link that cannot be sampled (native and
   in the browser lane) use one of -1e307 to 1e307 instead. The link allow-list also
   holds the layers and truncation its request carries, which it did not.
5. **A function's samples held to the slider.** Fixed. A file's surface is
   held to Swift's own bounds (`SurfaceRequest::check`, which reading the
   manifest applies: the function's samples to 4096, curves of at most
   2^24 vertices, the fit, the harmonics), not the sliders', so a bundle
   `kurven-cli surface sn --grid 2048` writes opens (tested natively and in
   the browser lane with the CLI's own bundle), as does a torus with 300 by
   150 coordinate curves. The same mismatch held a file's hatchings and
   slope lines to a slider's 400 across, where kurven-cli writes finer
   (`--spacing`, `--seeds`): a file is held to a bound on work instead,
   4000 strokes and 2000 seeds across (`files::FILE`), a link still to the
   sliders (`files::LINK`).
6. **The status line's name.** Fixed: "opened gamma.kurven", the bundle's
   own name (tested).
7. **Two copies of the bounds.** Fixed: `files` holds the one set of checks
   with `Limits` (`LINK`, `FILE`); `linkable` keeps only the link's own
   (resolution, level counts, which kinds a link carries) and calls
   `files::window`, `files::truncation` and `files::layers`.
8. **The document builder.** `Document::successor` (3).
9. **faces_now.** Fixed: one memo (`faces_memo`) behind `faces_of` and
   `faces_now`, each with its own comment.
10. **Convergents per reply.** Fixed: `Rests` reads the forced torus's own
    slope and its convergents once, and every row, the detents and the
    description read them from it.
