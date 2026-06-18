"""Perimeter: a domain-space boundary outline, specified once, from which the
occluder wall curtains and the scaffold ink (wall hatch, and — later — the
ground polygon and corner posts) all derive.

This is the keystone that removes the triplication the examples carry today: the
same cutout corners get hand-listed three times — once as `wall_curtain` calls
for the Z-buffer occluder, once as the ground-polygon segments, once as the
hatch edges. A `Perimeter` holds the edges once; the occluder mesh and the ink
are both views of it.

It bridges the two sibling modules — `occluder` (the hidden mesh) and `scaffold`
(the visible ink) — so it imports from both rather than living in either.
"""

import numpy as np

from kurven.occluder import wall_curtain
from kurven.scaffold import wall_hatch


class Edge:
    """A straight boundary segment from corner `start` to `end`, each an
    `(im, re)` domain point. Sampling linspaces between the endpoints in this
    direction, so every consumer (occluder curtain, ink hatch) samples the same
    points in the same order."""

    def __init__(self, start, end):
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))

    def samples(self, density):
        """`(im, re)` arrays of `density` points along the edge."""
        im = np.linspace(self.start[0], self.end[0], density)
        re = np.linspace(self.start[1], self.end[1], density)
        return im, re

    def wall_curtain(self, surface, density, *, base=0.0):
        """The occluder mesh curtain for this edge (see `occluder.wall_curtain`)."""
        im, re = self.samples(density)
        return wall_curtain(im, re, surface, base=base)

    def wall_hatch(self, surface, project, density, *, trim=False,
                   base=0.0, top_offset=0.0):
        """The ink hatch strokes for this edge (see `scaffold.wall_hatch`).

        `trim` drops the first and last sample (`[1:-1]`) so adjacent edges'
        hatch strokes don't double up at shared corners."""
        im, re = self.samples(density)
        if trim:
            im, re = im[1:-1], re[1:-1]
        return wall_hatch(im, re, surface, project, base=base, top_offset=top_offset)


class Perimeter:
    """An ordered list of boundary `Edge`s outlining a cutout (or the whole
    rectangular domain). Specify the edges once; derive the occluder wall
    curtains and the ink from the same definition."""

    def __init__(self, edges):
        self.edges = [e if isinstance(e, Edge) else Edge(*e) for e in edges]

    @classmethod
    def rectangle(cls, im_bounds, re_bounds):
        """The four edges of the rectangular domain `im_bounds × re_bounds`, in
        the order front, back, left, right (front = the low-imag/real-axis edge,
        sampled along increasing real)."""
        (i0, i1), (r0, r1) = im_bounds, re_bounds
        return cls([
            Edge((i0, r0), (i0, r1)),   # front (real axis, im = i0)
            Edge((i1, r0), (i1, r1)),   # back  (im = i1)
            Edge((i0, r0), (i1, r0)),   # left  (re = r0)
            Edge((i0, r1), (i1, r1)),   # right (re = r1)
        ])

    def wall_curtains(self, surface, density, *, base=0.0):
        """One occluder curtain per edge — pass straight to
        `build_occluder(..., walls=...)`. `density` is either an int (same for
        every edge) or a per-edge sequence (e.g. longer edges sampled denser)."""
        if isinstance(density, (int, np.integer)):
            density = [density] * len(self.edges)
        return [e.wall_curtain(surface, d, base=base)
                for e, d in zip(self.edges, density)]

    def ground_polygon(self, project, *, z=0.0):
        """Each edge as a straight ground line at height `z`, projected to 2D —
        the outline the cutout casts on the base plane. Returns a list of (2, 2)
        segments (one per edge), the same corners the wall curtains rise from."""
        out = []
        for e in self.edges:
            seg = np.array([[e.start[0], e.start[1], z],
                            [e.end[0], e.end[1], z]])
            out.append(project(seg)[:, :2])
        return out
