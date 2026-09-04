"""Scaffold: the visible structural line-work that frames a landscape plate —
the drawn (ink) counterpart to the hidden `occluder` mesh.

`occluder.py` meshes the surfaces that *block* sight lines (the Z-buffer
geometry). This module draws the surfaces and edges you actually *see* around
and beneath the contour field: the vertical hatch strokes down a cut face, and
(future) the ground-polygon perimeter, corner posts, and plateau-cap hatching.
They are duals — same perimeter, same `Surface`, same projection — so
`wall_hatch` deliberately mirrors `occluder.wall_curtain`'s signature: one emits
triangles for the depth buffer, the other emits strokes for the SVG.
"""

import numpy as np
def wall_hatch_3d(im, re, surface, *, base=0.0, top_offset=0.0):
    """`wall_hatch` before the camera: the same strokes as `(2, 3)` segments in
    `(im, re, z)`.

    The hatch geometry is camera-independent — a vertical segment at each
    perimeter sample — so it belongs in a `Scene` (and in a `.kurven` bundle)
    alongside the contours, not in the drawing code. `wall_hatch` is now this
    plus a projection, so the two cannot drift.
    """
    im = np.asarray(im, dtype=float)
    re = np.asarray(re, dtype=float)
    crest = surface.height_at(re, im) + top_offset
    return [np.array([[i, r, base], [i, r, h]])
            for i, r, h in zip(im, re, crest)]


def wall_hatch(im, re, surface, project, *, base=0.0, top_offset=0.0):
    """Vertical hatch strokes along a perimeter polyline, projected to 2D — the
    ink-side twin of `occluder.wall_curtain`.

    For each perimeter sample, draw a vertical segment from `z = base` up to the
    clamped surface height (`Surface.height_at`) plus `top_offset`, then project
    it to 2D. One stroke per sample.

    im, re:      equal-length perimeter samples in domain coords (im = imag,
                 re = real), matching `wall_curtain`.
    surface:     a `Surface`; `height_at(re, im)` gives the (clamped) crest z.
    project:     callable mapping an (N, 3) array of (im, re, z) rows to
                 projected coordinates; only the first two output columns are
                 kept. Pass the example's projection closure.
    base:        floor z each stroke rises from (a small epsilon lifts the ink
                 off the ground plane).
    top_offset:  added to the crest height (a small negative value tucks the
                 stroke just under the surface silhouette).

    Returns a list of (2, 2) projected segments, ready for `ax.plot(*seg.T)`.
    """
    return [project(seg)[:, :2]
            for seg in wall_hatch_3d(im, re, surface, base=base,
                                     top_offset=top_offset)]
