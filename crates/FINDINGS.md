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
