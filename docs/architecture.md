# How kurven draws a plate

A plate is the modulus surface of a meromorphic function f, the graph of
|f(z)| over a rectangle of the complex plane, drawn with the level curves of
|f| and arg f on it and every hidden part left out. The surface is truncated
at a height h(x), constant on bands in Re z, which the code calls the *cap*.
kurven gets there in seven stages:

**sample → contour → refine → lift → project → judge → ink**

The first four do not depend on the camera and run once per function; the
last three run again for every view (see *[The camera seam](#the-camera-seam)*).

## 1. Sample

f is evaluated on a uniform grid over the window (`NativeLandscape.build`,
across the cores). The grid is not the surface. It decides which contours
exist and brackets where they run; evaluations of f decide where they
actually are.

## 2. Contour

Marching squares over the grid, for |f| and for arg f, each at major and
minor levels (`Contour`, in scan order, so a grid gives the same lines every
run). A phase grid wraps at ±π, and marching squares emits a crossing for
every level along the wrap line; a vertex whose two edge samples differ by
more than π is on a wrap, not a level curve, and is dropped.

## 3. Refine

Let g be the residual along a grid edge: |f| − c for a modulus line,
arg f − θ for a phase line. Marching squares puts a vertex where linear
interpolation between the edge's two samples crosses zero, which is off the
true level curve by about h²·|g″|/|g′|: nothing where g is nearly linear
across a cell, and growing like 1/r toward a pole or a zero, exactly where
the plates are densest. So every vertex is moved to the exact zero of g on
its edge by a bracketed root find, and every chord between two vertices is
checked at its midpoint and subdivided until it is within 0.02 cells of the
level curve (`ContourRefine`).
Density ends up proportional to curvature, with no seams to stitch.

Near a zero the lattice cannot bracket everything. Rings around a zero that
no lattice node sees are seeded from the zero itself; a phase line whose end
the lattice could not bracket is followed by evaluating f, step by step
along its tangent with Newton's method back onto the level curve, until it reaches the window's
edge, a zero, or a point where the phase is undefined; a chord the refiner
cannot place on the level curve is not drawn. The details are in
[landscapes.md](landscapes.md#placing-contours-exactly).

## 4. Lift

Each vertex (x, y) is lifted to (x, y, |f(x + iy)|). When the function is at
hand, everything that is drawn or judged comes from evaluating f, not from
the grid: the height of the ink,
the surface normal, the fold curves, the wall's crest, the rim. The
lattice only brackets. Without the function (a bundle from a frozen plate),
the lattice is the surface.

Where |f| exceeds the truncation height the lift is clamped to h(x), and a
run that reaches it is carried to the rim, the curve |f| = h(x), solved
exactly. The walls (the solid's vertical sides) and the plateaus are hatched:
`wallHatch` up the walls, `wallOutline` for each wall's crest and foot,
`capHatch` across each plateau, `capOutline` for the rim, which under band
caps is not a level curve of |f| at all.

## 5. Project

The plates are **oblique parallel projections**. The imaginary axis is sheared
along the real one in the ground plane, then a rotation about z and a tilt
about x turn the surface toward the viewer. The real axis lands horizontal on
the paper, the |f| axis vertical and at true length, and the imaginary axis
recedes at a slant. Each plate has its own slant; measured on the paper,
relative to the vertical:

| plate | real axis | imaginary axis |
|---|---|---|
| Γ, 1/Γ | 1.22 | 0.93 at 49° |
| cn | 1.12 | 0.77 at 42° |
| ζ | 0.76 | 0.26 at 135° |

A textbook cabinet projection is 1.00 and 0.50 at 45°, cavalier 1.00 and 1.00
at 45°; the plates sit between them, with the real axis stretched, which is
the look no isometric drawing has. The three numbers a plate stores (`shear`,
`xAngle`, `yScale`, with `zAngle` always −90) are that family in other
coordinates: with the turn fixed, any receding angle, receding length and
real-axis length is reachable.

The shear is applied to the ground plane before the camera turns
(`PlateStyle.matrix`). With the front face head-on (turn ±90°) that is exactly
an oblique projection. From any other turn it is a sheared solid seen
orthographically: a round section draws as an ellipse with axes in the ratio
1.64. An orbit that leaves the plate's turn should set the shear to zero.

## 6. Judge

A heightfield bounds a solid: the region under the truncated graph, down to
the floor, closed over the window's edges by vertical walls. Let d be the
projection direction, pointing away from the viewer, and call the line
through a point parallel to d its sight line. A point of ink is hidden when
the solid meets its sight line between it and the viewer. Every clipped
layer is judged per vertex by two tests (`Scene.judgedLayers`):

- **Facing.** A point where the surface faces away from the viewer
  (n · d > 0) has a sight line that has already entered the solid, so it is
  hidden. With the function at hand the normal is the true surface's, from
  finite differences of |f| at a step small against the cell, so ink placed
  on the true surface is judged against it.
- **The march** (`SightMarch`). Along the sight line, the height of the
  lattice's surface is linear between the points where the line crosses a
  lattice row, a lattice column or a cell's diagonal, so its least clearance
  lies at one of them. The march visits those crossings, the rim crossings
  and the band boundaries, and no other points. With the function at hand it
  reads f at the crossings and searches each interval between them for the
  point where the line comes closest to the surface. Nothing is sampled: a
  sliver thinner than any step is still a crossing.

The margin means two things, kept apart. A point of ink lies on the modulus
surface, and the drawn facets miss that surface by an interpolation gap, so the point may start a little
inside the solid: that is forgiven perpendicular to the facet. Once the sight
line has been outside the solid, anything it enters is an occluder, judged
along the line.

The answer is a function of the geometry and the camera alone. A bake with a
64-pixel depth buffer and one with a 4096-pixel buffer are byte-identical.

Ink that depends on the camera is derived for it. **Fold curves**
(`foldLines`) are where d is tangent to the surface, n · d = 0. Along them
the projection of the surface onto the paper folds over, and their image is
the outline, outer and inner. d spans the kernel of the camera's first two
rows (`sightLine`), which under the plates' shear is not the gradient of
depth; using the gradient would tilt every fold. The facing is sampled on the lattice, its zero traced by marching squares,
each vertex bisected onto the exact fold, a run that ends beside a zero
carried to it, and a run that climbs a flank ended at its own crossing of the
rim. A fold inside a pit is never drawn, because its tangent line runs through
the bowl's own material. **Wall ink** is judged by the wall it lies in: a
wall that faces away hides its hatch, and the crest, which is on the surface,
is judged by the march alone.

**The depth buffer** is still drawn, and the realtime preview uses it: the
preview tests each line fragment against the depth pass, with a margin scaled
by the surface's slope as shadow maps bias by slope, so ink on a steep flank
does not crawl as the camera moves. The bake's depth output and raster outline
come from it too. The judge does not. The depth pass is held to an exact ray
cast of the capped heightfield in the test suite.

## 7. Ink

The judged runs, projected, are written as SVG polylines, one group per
layer, with stroke widths in plate units for a sixteen-inch page.

## The camera seam

The pipeline divides at the camera, not between library and application.
Sampling, contouring, placing and lifting do not change when the view does;
fold lines, judging and projecting do. The first half is the expensive one,
and its result is a `Scene`.

A **`.kurven` bundle** is a `Scene`, serialized: a directory holding a typed
`manifest.json` and `.npy` arrays. A landscape's bundle is described all the
way through: no vertices, a manifest and two grids (|f| unclamped, and arg f),
a few hundred kilobytes. Its contour layers are statements of levels, its
walls a perimeter, its truncation a value, so each is a control in the app.
The published plates' bundles, written by their Python programs, carry some
layers as dumped arrays, and the schema says which.

Bundle arrays are in world order, `x = real`, `y = imag`, `z = |f|`. The
Python library carries `(imag, real, z)` internally; the exchange happens in
one function (`kurven.bundle.swap_to_world`), and the convention is written
into the manifest, because forgetting it is the single most common bug in
this codebase's history.

GPU resources are keyed on `Scene.content`, which survives a camera change, so
navigation costs one uniform upload plus the depth pass: 1.6 ms for recip and
5.9 ms for zeta at 1024². A full preview frame at 3200² is 2.2 ms for recip,
6.8 ms for elliptic and 12.1 ms for zeta (`kurven-cli bench`). A bake is a
march per vertex and is slower: one to twenty-five seconds for the plates in
the README.

## The app

```bash
scripts/bundle-app.sh                    # assembles build/Kurven.app
open -a build/Kurven.app recip.kurven    # or double-click the bundle
swift run -c release KurvenApp --open ../recip.kurven    # from a terminal
```

Left-drag orbits, shift-drag pans, scroll zooms toward the cursor,
double-click re-targets the turn onto the point you clicked, `f` fits,
`1`/`2`/`3` switch between the plate, a shaded surface, and the raw depth
buffer. The inspector carries the camera as numbers, the plate presets,
per-layer visibility and levels, the hidden-line margin, and a bake panel.

⌘N opens a picker: the catalog as pictures, each cell the landscape it names,
drawn small by the same renderer that draws it full size. A name cannot answer
what ψ looks like against ζ, which is the reason these plates were drawn in
1909 rather than tabulated. They cost about 30 ms each and are cached under
`~/Library/Caches/world.kurven`; `Kurven --thumbnails` draws them all.

The sidebar's top section is the function (a catalog dropdown, `Browse…` for
the same gallery, and a field that takes any expression the language parses),
the window as two thumbs per axis, the resolution, and the truncation: a
height, or a staircase of bands in Re. A described layer carries its own
control: a level count for a contour family, a spacing for a hatching.
Dragging a window slider samples at a draft resolution and at the full one
when you let go; dragging the truncation height does not sample at all (see *[Two speeds of
edit](landscapes.md#two-speeds-of-edit)*). ⌘S writes what is on screen as a
`.kurven` bundle.

Navigation is a pure function: input handling produces `Gesture` values and
`Navigator.applying` folds them, so orbit, pan, zoom and re-target are tested
without a window. The app is scriptable for the same reason:

```bash
Kurven.app/Contents/MacOS/Kurven recip.kurven --screenshot out.png
Kurven.app/Contents/MacOS/Kurven recip.kurven --bake out.svg --resolution 4000
Kurven.app/Contents/MacOS/Kurven --landscape zeta --resolution 800 \
    --cap 4 --save zeta.kurven --screenshot zeta.png
```

The second is how "the app bakes what the CLI bakes" is checked: it is the
same `Scene` value through the same function, and the two SVGs are
byte-identical.

## Surfaces and solids

`kurven-cli surface` bakes ink on parametric surfaces, where a point's
visibility is decided by where on the surface it lies: the torus's parameter
lines, a doubly periodic function's contours on the torus its period
rectangle glues into (`sn`, or any expression with `periodic`), a forced
system's invariant torus with its trajectory, and polyhedra
(`surface solid`: the Platonic solids and Császár's seven-vertex torus),
whose edges are judged by a sign rule and a closed-form cast against every
face. Run `kurven-cli` with no arguments for the options.
