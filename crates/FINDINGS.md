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
