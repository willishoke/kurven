# Landscapes you choose

A published plate was a program. The gamma plate knew that it was drawing Γ,
on [−4.5, 4.5] × [−2.5, 2.5], with four pole spires truncated at four different
heights, hatched down the front face at a density someone picked by eye. That
is the right way to reproduce a plate from 1909 and the wrong way to look at a
function nobody has drawn yet.

So the four facts that make a landscape (**which function, over what window,
truncated where, sampled how finely**) are a value, and everything else is
derived from them by rules stated once.

```bash
kurven-cli landscape "exp(1/z)" --re -1.2,1.2 --im -1.2,1.2 --truncation 5 -o mine.kurven
kurven-cli landscape --function zeta --res 800 -o zeta.kurven     # a catalog preset
kurven-cli bake mine.kurven -o mine.svg
```

## The expression

A small complex expression language with its own tokenizer and
recursive-descent parser:

```
gamma(z)      1/Γ(z)        zeta(z)        exp(1/z)      z^3 - 1
tan(z)        sin(z)cn(z, 0.64)            (z^2-1)/(z^2+1)          z!
```

Implicit multiplication (`2z`, `(z-1)(z+1)`), `^` for powers, postfix `!` for
Γ(z+1), unicode aliases (`Γ`, `ζ`, `ψ`, `π`), and 43 functions
(`kurven-cli catalog` lists them). Errors carry the character they went wrong
at, which is what lets the app's expression field say where.

The language and its functions are implemented in Swift with no library behind
them (`Sources/KurvenMath`): complex arithmetic on numpy's
branches, scipy's own algorithms where scipy has one (Hare's log-gamma,
cephes' `ellpj`, the Lambert W iteration), Weideman's Faddeeva function for
the error functions, and for the Bessel and Airy families two engines, the
power series and the Hankel expansions for J and Y, Temme's series and
Steed's continued fraction for I and K, each used only where it is accurate.
Two functions have no complex form in scipy and are implemented from the
literature: **ζ(s)** (Borwein's alternating series, the functional equation
below the critical line, and Euler–Maclaurin at the points where Borwein's
own `1 − 2^(1−s)` factor vanishes: σ = 1, t ≈ 9.06, 18.13, …, which are
inside the ζ plate's own window), and the **Jacobi elliptics** sn, cn, dn.

`tests/fixtures/expr` holds every function to an oracle at every point of a
grid that runs along the real axis at exactly `im = 0`: numpy for the
elementary functions, scipy.special where it has a complex form, and mpmath
for ζ and the Jacobi three, to 1e-12, or 1e-9 for Bessel and Airy, whose
engines differ from AMOS. The language itself was ported from a Python
original, since retired; the readings two reasonable people disagree about
(`-z^2`, `2z^2`, `1/2z`) are frozen in `tests/fixtures/expr/language.json`.

Every number written in the expression is also a control. `cn(z, 0.64)` puts a
row under the field labelled *Modulus*, `besselj(2, z)` one labelled *Order*,
`z^3` one labelled *Exponent*: a slider, a value field and a stepper, with
undo. Moving one rewrites that literal in the expression and resamples, so the
field reads `cn(z, 0.71)` because that is what the landscape now is. The
expression is the whole state, after a drag as before it.

The catalog (`Catalog.native`, `kurven-cli catalog`) is fourteen presets over
that language, each an expression plus the window and truncation that make it
read as a landscape.

## Placing contours exactly

Let g be the residual along a grid edge: |f| − c for a modulus line,
arg f − θ for a phase line. Marching squares puts a vertex where *linear
interpolation* between the edge's two samples crosses zero, which is off the
true level curve by about h²·|g″|/|g′|. The refiner (`ContourRefine`) moves
every vertex to the exact zero of g on its edge by a bracketed root find,
evaluating f itself rather than the grid, and checks every chord between
two vertices at its midpoint, subdividing along the normal until it is
within 0.02 cells of the level curve. Where a level
passes through a critical point of f (|cn| = 1 at 0, |sin| = 1 at π/2) the
level curve crosses itself, marching squares draws two arcs a cell apart, and
the solve along the normal meets a double root; the refiner accepts its
nearest iterate there and draws the crossing. The grid still decides which
contours exist; evaluations of f decide where they run.

Near a zero, three more rules apply:

- **Rings the grid cannot see.** A zero found from a grid minimum seeds
  the rings around it when no grid node falls inside them.
- **Phase lines past the bracket.** Every phase line meets at a zero, and the
  grid loses them a cell or two short. An open end inside the window is
  followed by evaluating f: a step along the tangent, Newton back onto the level curve,
  until the line reaches the window's edge, the zero, or a point where the
  phase is undefined (f underflowed or is not finite).
- **No chord the refiner cannot place.** A chord whose midpoint cannot be put
  on the level curve (a non-finite value, a quarter point on the wrong phase
  branch, the depth limit) breaks the run instead of being drawn across.

`kurven-cli refine --function gamma` benchmarks it: for each contour layer,
the grid's curves and the refined curves, timed, and the distance of each from
the true level curve (the residual over the gradient, in cells). At 600 samples across, gamma's
magnitude vertices go from a worst error of 0.58 cells to zero and its chords
from 0.16 to 0.02. The bake, the preview and the app refine whenever the
bundle names a function; `--no-refine` draws the grid's contours, which is
the stage the fixtures pin.

## Truncation, and the hatching that goes with it

A pole makes a landscape unboundedly tall, and a picture of one is a needle and
a floor. So the surface is truncated: the drawn surface is min(|f|, h(x)) for
a truncation height h. `Truncation` sets it: none, one height, or (gamma's
case) a different height per band of Re z, so each spire's plateau reads from
above.

The walls and the plateaus are then hatched, which is the part every
published plate wrote by hand, with hand-found spire radii and `while` loops
stepping outward until |f| fell under the limit. Here it is three described
layer kinds:

| kind | what it draws |
|---|---|
| `hatching` on `walls` | vertical hatch lines up the walls, every `spacing` along each perimeter edge, from the ground to the truncated surface |
| `hatching` on `plateaus` | hatch lines along an axis across each plateau, at height h(x), ending exactly on the truncation edge |
| `wallEdges` | each wall's top and bottom edges, and a vertical edge at each corner |
| `truncationEdge` | where the graph meets the plateaus: the curve \|f\| = h(x), which under a banded truncation is not a level set of \|f\| at all |

The hatchings are the same `Hatching` value that hatches a polyhedron's faces
([solids.md](solids.md)), laid on different faces.

## Two speeds of edit

Because those layers are *descriptions* rather than stored curves, changing
your mind costs one of two very different things:

- **The function, the window, the resolution** need f evaluated again. That is a
  new landscape (`NativeLandscape.build`, sampled across the cores) and a new
  bundle: about 10 ms at 600², 70 ms for ζ.
- **The truncation height, the contour levels, the hatch spacing** need nothing but the grids
  already in memory. `KurvenBundle.restyled` re-derives the curves, the wall
  curtains and the plateaus, and the picture follows within a frame.

Moving the truncation slider therefore moves the heightfield, the top edge
every wall's hatch line rises to, every plateau hatch line and the truncation
edge, and which
contours are truncated, all from one assignment.
