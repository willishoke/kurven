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


def wall_hatch(im, re, surface, project, *, base=0.0, top_offset=0.0):
    """Vertical hatch strokes along a perimeter polyline — the ink-side twin of
    `occluder.wall_curtain`.

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
    im = np.asarray(im, dtype=float)
    re = np.asarray(re, dtype=float)
    crest = surface.height_at(re, im) + top_offset
    out = []
    for i, r, h in zip(im, re, crest):
        seg = np.array([[i, r, base], [i, r, h]])
        out.append(project(seg)[:, :2])
    return out
