# Glossary

The words kurven uses for what it draws, and where each comes from. A name
is decided here first: a new term enters this file in the commit that
introduces it, after looking up what the field already calls the thing.
`scripts/check.sh` fails if a retired word comes back into `Sources/` or
the docs (`scripts/retired-words.txt`).

Most of the vocabulary comes from four places: boundary representation of
solids (Mäntylä, *An Introduction to Solid Modeling*, 1988), the
differential geometry of surfaces and their projections (do Carmo;
Koenderink, *Solid Shape*, 1990), hidden-line removal (Appel 1967;
Amanatides and Woo 1987; Hart 1996), and the drawings themselves (Jahnke
and Emde, *Funktionentafeln*, 1909).

## Geometry

| Term | Meaning | Was |
|---|---|---|
| landscape | The graph of \|f\| for a meromorphic f over a rectangle of the complex plane, truncated and closed into a solid. Kasner's *analytic landscape*. | |
| graph | The part of a landscape's surface where z = \|f(x + iy)\|. | flank |
| truncation | The cut that turns poles into plateaus: the drawn surface is min(\|f\|, h(x)), with the *truncation height* h constant on bands of Re z (`Truncation`). | cap |
| plateau | The flat part of a landscape where \|f\| ≥ h(x). | cap, cap face, truncated top |
| truncation edge | Where the graph meets a plateau: the level set \|f\| = h(x). | rim, cap outline |
| wall | A vertical face along an edge of the window's perimeter, closing the solid. Its *top edge* follows the surface, its *bottom edge* is on the floor, and a *vertical edge* stands at each corner. | crest, foot, post |
| crease | A curve where a surface is continuous but its normal is not, such as a truncation edge. Standard in line drawing. | |
| face | A face in the boundary-representation sense: a maximal flat region of a polyhedron's boundary, possibly not convex and possibly with holes. | facet |
| loop | A closed chain of a face's edges, one per component of its boundary: its *outer loop*, counter-clockwise seen from outside, and one loop per hole, clockwise. Where a face touches itself at a vertex, the loop there passes the vertex twice: a hole touching the outer loop is part of it, and two holes touching are one. A solid is given as its faces' loops (`Solid(vertices:faces:)`). | |
| convex piece | One polygon of a face's convex partition, which the solid computes from the face's loops (`ConvexPartition`). | face |
| flatness tolerance | How far a vertex may lie from its face's plane and the face still count as flat: a billionth of the face's *extent*, the diagonal of its bounding box in its plane, and never less than the rounding its coordinates carry (`Solid.FaceFrame.flatnessTolerance`). Standard in geometric tolerancing. | |
| diagonal | A side shared by two pieces of one face. Not an edge of the solid, and never drawn. | soft edge |
| edge | A side shared by two pieces of different faces: an edge of the solid. | crease, in the polyhedron code |
| parametric surface | A map from a parameter rectangle, each axis possibly periodic, into space (`ParametricSurface`). | |
| grid | The uniform sample grid a surface or a function is evaluated on. | lattice |
| lattice | A lattice in the mathematical sense: the period lattice of an elliptic function (`PeriodLattice`), Turing's square lattice. | |
| seam | Where a periodic chart's rectangle is glued to itself. | also tile and chunk boundaries |
| tubular neighbourhood | The coordinates of a point near a surface: the foot of its normal and its signed distance along it (`TubularNeighbourhood`). | enclosure |
| normal offset | How far a surface stands off its base surface along the base's normal. | standoff |

## Curves

| Term | Meaning | Was |
|---|---|---|
| curve | A polyline on a surface, in world space, before visibility and projection. | ink |
| level set, contour | The curve where a field takes one value: \|f\| = c, arg f = θ. | |
| restriction | The part of the domain a contour is kept to: the footprint, below the truncation, a band, or an intersection of these (`Restriction`). | keep |
| coordinate curves | The curves u = const and v = const on a parametric surface. | parameter lines |
| linear curves | Straight lines through a torus's parameter rectangle; a (p, q) torus curve when the slope p/q is rational. | windings |
| fold curve | Where the projection direction is tangent to the surface: Whitney's fold of the projection. Its image on the page is the *outline*. | fold lines |
| outline | The image of the fold curves. | silhouette |
| hatching | Parallel lines at an angle and a spacing, laid on faces (`Hatching`, `HatchedFaces`). Each one is a *hatch line*. | fill, ruling |
| tone | The function on a face that sets where hatch lines are drawn and how wide: `.uniform`, `.turing`, `.halfLambert`. | fill field |
| Turing's pattern | The closed form behind the dappled figure in Turing's 1952 paper, as reconstructed in "Turing Patterns" (willishoke.github.io, 2022): unit Gaussians on a square lattice of spacing 0.7 with random signs. Not a reaction–diffusion simulation. | |
| half-Lambert | The tone (1 − n·l)/2 for a face's normal n and the unit direction l toward the light: the darkness under half-Lambert, or wrapped, shading (1 + n·l)/2 (Mitchell, McTaggart and Green, "Shading in Valve's Source Engine", 2006). Not Lambert's cosine law, max(0, n·l). | lambert, shade |
| width steps | How many widths a hatch line can be drawn at. | levels |
| step | The distance between successive vertices, or readings of a tone, along a curve. | pitch |

## Visibility

| Term | Meaning | Was |
|---|---|---|
| visibility | Whether a point of a curve is seen: it lies on a face turned toward the eye, and nothing stands between it and the eye. | judging |
| projection direction | The direction d the camera projects along, away from the viewer. A *line of sight* is the line through a point parallel to d. | sight line |
| grid traversal | The heightfield's occlusion test: visit every grid row, column and diagonal crossing along a line of sight. Amanatides and Woo's traversal. | march |
| sphere tracing | The displaced torus's occlusion test: step along a line of sight by the distance to the surface over its Lipschitz constant (Hart 1996). | march |
| occlusion | A polyhedron's occlusion test: the spans of a flat segment hidden by flat pieces, in closed form (`Solid.Occlusion`). | cast |
| depth tolerance | How far behind a surface, in view depth, a point may lie and still be drawn. The preview adds a slope-scaled term, as shadow maps do. | margin |

## Drawing

| Term | Meaning | Was |
|---|---|---|
| stroke | What is drawn on the page: a projected curve with a width. | ink |
| oblique projection | The parallel projection the plates use: a shear in the ground plane, then a turn and a tilt (`ObliqueProjection`, `Camera.oblique`). | plate projection |
| drawing convention | The part of an oblique projection that is not a rotation: the axis exchange, the flip, the y scale and the shear (`DrawingConvention`). | plate style |
| page space | 2D drawing coordinates, in SVG units (`PageSpace`). | plate space |
| drawing | What kurven produces for a scene: a set of strokes. A parametric surface's is a `SurfaceDrawing`. | plate |
| plate | One of Jahnke and Emde's 1909 plates, or kurven's reproduction of one. Nothing else. | |

## Kept, and why

- **fold**: Whitney's term for the critical set of a projection.
- **crease**: standard in line drawing for a curve of normal discontinuity
  on a surface, such as a truncation edge. On a polyhedron every such curve
  is an edge, and edge is the word there.
- **wall**: plain and unambiguous; "lateral face" is the formal term.
- **level**: the value of a contour, and only that.
- **trajectory**, **tile**, **window**, **occluder**, **footprint**, `Cover`: already standard.
- **margin**: the paper round a drawing on its page (`SVG.render`, the
  framing, the gallery). The hidden-line margin, which was a depth, is the
  depth tolerance.
- `sight`: the projection direction as a parameter name, where a vector is
  passed to a fold trace or a shader. In prose it is a line of sight.
- **spire, pit, bowl, needle**: describe how a pole or a zero looks in a
  drawing, in prose only, never in identifiers.
