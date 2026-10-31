# How kurven draws

kurven draws curves on the boundary of a solid, leaves out what the solid
hides, and writes what is left as strokes in an oblique projection. Three
kinds of solid share that pipeline:

- a **landscape**: the graph of |f| for a meromorphic f over a rectangle of
  the complex plane, truncated at a height h(x) and closed by walls into a
  solid (`Heightfield`);
- a **parametric surface**: a torus or another surface given by a map from a
  parameter rectangle (`ParametricSurface`);
- a **polyhedron**, bounded by flat faces (`Solid`).

For each, the same five steps: the solid's faces; the curves on them;
visibility; projection; strokes. Only the occlusion test differs between
the three, and it is the part that makes the drawing exact. The words used
here are defined in the [glossary](glossary.md).

## The model

**Faces.** A landscape's boundary is the graph, where z = |f(x + iy)|; the
plateaus, where |f| reaches the truncation height and the surface is cut
flat; and the walls along the window's edges. A torus is one face. A
polyhedron's faces are flat, and a face that is not convex is stored as
its convex pieces (see [solids.md](solids.md)).

**Curves.** Most curves are level sets of a field on a face: contours of
|f| and arg f on the graph, coordinate curves on a torus, hatch lines on a
wall or a plateau. The others are a face's edges (the truncation edge,
where the graph meets a plateau; a wall's top, bottom and vertical edges;
a polyhedron's edges), flow lines (a trajectory, the linear curves on a
torus), and fold curves, which depend on the camera. A curve is a polyline
in world space; on a parametric surface each vertex also carries its
surface coordinate.

**Visibility.** A point of a curve is seen when the face under it faces the
eye and nothing stands between it and the eye. The first test is a sign:
n · d < 0 for the outward normal n and the projection direction d, which
points away from the viewer. A point on a face turned away has a line of
sight that has already entered the solid. The second is an occlusion test,
one for each kind of solid:

| solid | occlusion test |
|---|---|
| landscape | grid traversal along the line of sight (`GridTraversal`) |
| torus, displaced torus | sphere tracing against the tubular neighbourhood (`SurfaceVisibility`) |
| other parametric surfaces | the coordinate image: the surface in front of the point must be the point's own neighbourhood of the surface |
| polyhedron | closed-form occlusion of a flat segment by flat pieces (`Solid.Occlusion`) |

Where visibility changes along a segment, the segment is cut there by
bisection on the exact test, not at a pixel. Every test but the coordinate
image has no resolution in it: a bake with a 64-pixel depth buffer and one
with a 4096-pixel buffer write the same strokes, byte for byte.

**Projection.** An oblique parallel projection (`Camera.oblique`), as the
plates use. See [below](#the-oblique-projection).

**Strokes.** The visible spans, projected, are written as SVG polylines,
one group per layer, with stroke widths in page units for a sixteen-inch
page.

## Before and after the camera

The pipeline divides at the camera. Sampling, contouring, placing curves
on the surface and lifting them do not change when the view does; fold
curves, visibility and projection do. The first half is the expensive one,
and its result is a `Scene`.

A **`.kurven` bundle** is a `Scene`, serialized: a directory holding a typed
`manifest.json` and `.npy` arrays (schema 2, `Manifest.swift`). A
landscape's bundle is described all the way through: no vertices, a
manifest and two grids (|f| untruncated, and arg f), a few hundred
kilobytes. Its contour layers are statements of levels, its walls a
perimeter, its truncation a value, so each is a control in the app. The
bundles of the published plates, written by their retired Python programs,
store some layers as arrays, and the schema says which.

Bundle arrays are in world order, `x = real`, `y = imag`, `z = |f|`. The
convention is written into the manifest, because forgetting it was the most
common bug in this codebase's history.

GPU resources are keyed on `Scene.content`, which survives a camera change,
so navigation costs one uniform upload plus the depth pass: 1.6 ms for recip
and 5.9 ms for zeta at 1024². A full preview frame at 3200² is 2.2 ms for
recip, 6.8 ms for elliptic and 12.1 ms for zeta (`kurven-cli bench`). A bake
runs the occlusion test per vertex and is slower: one to twenty-five seconds
for the plates in the README.

## Landscapes

**Sample.** f is evaluated on a uniform grid over the window
(`NativeLandscape.build`, across the cores). The grid is not the surface.
It decides which contours exist and brackets where they run; evaluations of
f decide where they actually are.

**Contour.** Marching squares over the grid, for |f| and for arg f, each at
major and minor levels (`Contour`, in scan order, so a grid gives the same
curves every run). A phase grid wraps at ±π, and marching squares emits a
crossing for every level along the wrap; a vertex whose two edge samples
differ by more than π is on the wrap, not on a level set, and is dropped.

**Refine.** Let g be the residual along a grid edge: |f| − c for a modulus
contour, arg f − θ for a phase contour. Marching squares puts a vertex where
linear interpolation between the edge's two samples crosses zero, which is
off the true level set by about h²·|g″|/|g′|: nothing where g is nearly
linear across a cell, and growing like 1/r toward a pole or a zero, exactly
where the plates are densest. So every vertex is moved to the exact zero of
g on its edge by a bracketed root find, and every chord between two vertices
is checked at its midpoint and subdivided until it is within 0.02 cells of
the level set (`ContourRefine`).

Near a zero the grid cannot bracket everything. Rings around a zero that no
grid node sees are seeded from the zero itself; a phase contour whose end
the grid could not bracket is followed by evaluating f, step by step along
its tangent with Newton's method back onto the level set, until it reaches
the window's edge, a zero, or a point where the phase is undefined; a chord
the refiner cannot place on the level set is not drawn. The details are in
[landscapes.md](landscapes.md#placing-contours-exactly).

**Lift.** Each vertex (x, y) is lifted to (x, y, |f(x + iy)|). When the
function is at hand, everything that is drawn or tested comes from
evaluating f, not from the grid: the height of a curve, the surface normal,
the fold curves, a wall's top edge, the truncation edge. The grid only
brackets. Without the function (a bundle from a frozen plate), the grid is
the surface.

**Truncation.** Where |f| exceeds the truncation height h(x) the surface is
cut flat. A contour that reaches a plateau is carried to the truncation
edge, the level set |f| = h(x), solved exactly. Under a banded truncation
(gamma's, one height per band of Re z) the truncation edge is not a level
set of |f| at all, and the step between two bands is a wall of its own.

**Hatching.** Walls and plateaus are hatched: vertical lines up the walls,
lines along an axis across the plateaus (`Hatch.swift`). Both are the same
`Hatching` value laid on different faces (`HatchedFaces`), and the spacing
is rounded on each face to a whole number of gaps, so a plateau's hatch
lines meet the walls' at a corner. `wallEdges` draws each wall's top,
bottom and vertical edges.

**Visibility.** The solid is the region under the truncated graph, down to
the floor, closed by the walls. Every clipped layer is tested per vertex
(`Scene.visibleLayers`):

- **Facing.** With the function at hand the normal is the true surface's,
  from finite differences of |f| at a step small against the cell, so a
  curve placed on the true surface is tested against it.
- **Grid traversal** (`GridTraversal`). Along the line of sight, the height
  of the grid's surface is linear between the points where the line crosses
  a grid row, a grid column or a cell's diagonal, so its least clearance
  lies at one of them. The traversal visits those crossings, the crossings
  of the truncation edge and the band boundaries, and no other points. With
  the function at hand it reads f at the crossings and searches each
  interval between them for the point where the line comes closest to the
  surface. Nothing is sampled: a sliver thinner than any step is still a
  crossing.

The depth tolerance means two things, kept apart. A point of a curve lies
on the modulus surface, and the drawn triangles miss that surface by an
interpolation gap, so the point may start a little inside the solid: that
is forgiven perpendicular to the triangle. Once the line of sight has been
outside the solid, anything it enters is an occluder, tested along the
line.

**Fold curves** (`foldCurves`) are where d is tangent to the surface,
n · d = 0. Along them the projection of the surface folds over, and their
image is the outline, outer and inner. d spans the kernel of the camera's
first two rows (`projectionDirection`), which under the plates' shear is
not the gradient of depth; using the gradient would tilt every fold. The
facing is sampled on the grid, its zero traced by marching squares, each
vertex bisected onto the exact fold, a run that ends beside a zero carried
to it, and a run that climbs the graph ended at its own crossing of the
truncation edge. A fold inside a pit is never drawn, because its tangent
line runs through the pit's own material.

**Curves on a wall** are tested by the wall they lie on: a wall that faces
away hides its hatching, and its top edge, which is on the surface, is
tested by the traversal alone.

**The depth buffer** is still drawn, and the realtime preview uses it: the
preview tests each stroke fragment against the depth pass, with the depth
tolerance scaled by the surface's slope as shadow maps bias by slope, so a
curve on a steep part of the graph does not crawl as the camera moves. The
bake's depth output and raster outline come from it too. Visibility does
not. The depth pass is held to an exact ray cast of the truncated
heightfield in the test suite.

## Parametric surfaces

A parametric surface is a map from a rectangle, each axis possibly glued to
itself, into space, with its derivatives (`ParametricSurface`). The map is
asked the questions; the grid it is sampled on is what gets rasterized. A
torus's curves are its coordinate curves and linear curves, a doubly
periodic function's contours on the torus its period rectangle glues into
(`sn`, or any expression with `periodic`), or a forced system's trajectory
on the invariant torus it winds round (`KurvenDynamics`). Each vertex
carries its surface coordinate, so visibility is asked where on the surface
it lies.

The facing test is the same sign. For occlusion there are two rules:

- **Sphere tracing.** The round torus and a torus displaced along its normal
  have a tubular neighbourhood in closed form: for any point near the
  surface, the foot of its normal and its signed distance along it
  (`TubularNeighbourhood`). Along the line of sight the distance to the
  solid changes by at most a known Lipschitz constant per unit of travel,
  so a step of that distance over the constant cannot pass through the
  solid (Hart 1996). The line is clear exactly when every step is.
- **The coordinate image.** Without a closed-form neighbourhood (the forced
  torus, a Fourier series in two angles) the surface is rasterized with its
  coordinates (`SurfaceImage`), and a point is seen when the surface in
  front of it, at its pixel or one beside it, is the point's own
  neighbourhood of the surface. A second sheet in front is far away in
  coordinates however close it is in depth, which is where a depth
  tolerance fails; but the raster is right only to its resolution.

Fold curves are traced as on a landscape: the facing sampled on the grid,
each vertex bisected onto the exact fold, and a fold that meets a crease
seeded from it.

## Polyhedra

A polyhedron is given as the convex pieces of its faces, counter-clockwise
from outside (`Solid(vertices:pieces:)`). Pieces of one plane that share a
side form a face, and the shared side is a diagonal of the partition, not
an edge: it is not drawn and hides nothing. The Platonic solids are built
from their vertex coordinates by a convex hull; Császár's seven-vertex torus
and a picture frame (a square annulus with depth, ten faces in sixteen
pieces) are the two that are not convex.

An edge is seen where one of its two faces faces the eye and no piece stands
in front. Occlusion is closed form: a segment and a piece both project to
flat regions of the page, their overlap along the segment is an interval,
and over it the difference of their view depths is linear, so the hidden
part is an interval solved, not searched. On a convex solid the sign test
alone decides; on a torus an edge can pass behind a piece elsewhere on the
body, and the occlusion test splits it.

Faces are hatched in their own plane: lines at an angle and a spacing whose
width is set by a tone (uniform, Turing's pattern, or the darkness under half-Lambert shading), each
line clipped to the face's pieces and joined across the diagonals. The
details and the picture frame are in [solids.md](solids.md).

## The oblique projection

The plates are **oblique parallel projections**. The imaginary axis is
sheared along the real one in the ground plane, then a rotation about z and
a tilt about x turn the surface toward the viewer. The real axis lands
horizontal on the page, the |f| axis vertical and at true length, and the
imaginary axis recedes at a slant. Each plate has its own slant; measured
on the page, relative to the vertical:

| plate | real axis | imaginary axis |
|---|---|---|
| Γ, 1/Γ | 1.22 | 0.93 at 49° |
| cn | 1.12 | 0.77 at 42° |
| ζ | 0.76 | 0.26 at 135° |

A textbook cabinet projection is 1.00 and 0.50 at 45°, cavalier 1.00 and
1.00 at 45°; the plates sit between them, with the real axis stretched,
which is the look no isometric drawing has. The three numbers a plate
stores (`shear`, `xAngle`, `yScale`, with `zAngle` always −90) are that
family in other coordinates: with the turn fixed, any receding angle,
receding length and real-axis length is reachable.

The shear is applied to the ground plane before the camera turns
(`DrawingConvention.matrix`). With the front face head-on (turn ±90°) that
is exactly an oblique projection. From any other turn it is a sheared solid
seen orthographically: a round section draws as an ellipse with axes in the
ratio 1.64. An orbit that leaves the plate's turn should set the shear to
zero.

## The app

```bash
scripts/bundle-app.sh                    # assembles build/Kurven.app
open -a build/Kurven.app recip.kurven    # or double-click the bundle
swift run -c release KurvenApp --open ../recip.kurven    # from a terminal
```

Left-drag orbits, shift-drag pans, scroll zooms toward the cursor,
double-click re-targets the turn onto the point you clicked, `f` fits,
`1`/`2`/`3` switch between the drawing, a shaded surface, and the raw depth
buffer. The inspector carries the camera as numbers, the bundle's camera
presets, per-layer visibility and levels, the depth tolerance, and a bake
panel. Polyhedra are not in the app yet; `kurven-cli surface solid` draws
them.

⌘N opens a picker: the catalog as pictures, each cell the landscape it
names, drawn small by the same renderer that draws it full size. A name
cannot answer what ψ looks like against ζ, which is the reason these plates
were drawn in 1909 rather than tabulated. They cost about 30 ms each and are
cached under `~/Library/Caches/world.kurven`; `Kurven --thumbnails` draws
them all.

The sidebar's top section is the function (a catalog dropdown, `Browse…` for
the same gallery, and a field that takes any expression the language
parses), the window as two thumbs per axis, the resolution, and the
truncation: a height, or a staircase of bands in Re. A described layer
carries its own control: a level count for a contour family, a spacing for
a hatching. Dragging a window slider samples at a draft resolution and at
the full one when you let go; dragging the truncation height does not
sample at all (see *[Two speeds of edit](landscapes.md#two-speeds-of-edit)*).
⌘S writes what is on screen as a `.kurven` bundle.

Navigation is a pure function: input handling produces `Gesture` values and
`Navigator.applying` folds them, so orbit, pan, zoom and re-target are tested
without a window. The app is scriptable for the same reason:

```bash
Kurven.app/Contents/MacOS/Kurven recip.kurven --screenshot out.png
Kurven.app/Contents/MacOS/Kurven recip.kurven --bake out.svg --resolution 4000
Kurven.app/Contents/MacOS/Kurven --landscape zeta --resolution 800 \
    --truncation 4 --save zeta.kurven --screenshot zeta.png
```

The second is how "the app bakes what the CLI bakes" is checked: it is the
same `Scene` value through the same function, and the two SVGs are
byte-identical.
