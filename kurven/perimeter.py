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
from kurven.scaffold import wall_hatch, wall_hatch_3d


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

    def wall_hatch_3d(self, surface, density, *, trim=False,
                      base=0.0, top_offset=0.0):
        """The ink hatch strokes for this edge as 3D `(2, 3)` segments (see
        `scaffold.wall_hatch_3d`) — the camera-independent form, which is what a
        `Scene` carries.

        `trim` drops the first and last sample (`[1:-1]`) so adjacent edges'
        hatch strokes don't double up at shared corners."""
        im, re = self.samples(density)
        if trim:
            im, re = im[1:-1], re[1:-1]
        return wall_hatch_3d(im, re, surface, base=base, top_offset=top_offset)

    def wall_hatch(self, surface, project, density, *, trim=False,
                   base=0.0, top_offset=0.0):
        """`wall_hatch_3d` projected to 2D (see `scaffold.wall_hatch`)."""
        return [project(seg)[:, :2]
                for seg in self.wall_hatch_3d(surface, density, trim=trim,
                                              base=base, top_offset=top_offset)]


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

    def is_closed(self, tol=1e-9):
        """True when the edges form a traversal — each edge's end is the next
        one's start, and the last closes onto the first.

        Not every `Perimeter` is one. `rectangle()` emits front/back/left/right
        because that is the order its consumers want to index (the front edge is
        `edges[0]`), which is a set of walls, not a loop. Only a traversal has an
        interior, so only a traversal can answer `contains`.
        """
        n = len(self.edges)
        return n >= 3 and all(
            abs(self.edges[k].end[0] - self.edges[(k + 1) % n].start[0]) <= tol
            and abs(self.edges[k].end[1] - self.edges[(k + 1) % n].start[1]) <= tol
            for k in range(n))

    def corners(self):
        """The polygon's vertices, `(im, re)`, one per edge start."""
        if not self.is_closed():
            raise ValueError(
                "perimeter is not a closed traversal; its edges have no interior "
                "(Perimeter.rectangle() is a set of walls, not a loop)")
        return np.array([e.start for e in self.edges], dtype=float)

    def contains(self, im, re):
        """Even-odd point-in-polygon over the closed perimeter. Broadcasts.

        This is what makes a cutout one definition rather than three: the walls,
        the ground ink, and the region the heightfield occluder is meshed over
        all come from these same corners, instead of a hand-written predicate
        that can drift from the polygon it is supposed to describe (zeta's did,
        by one unit in imag).
        """

        from kurven.bundle import point_in_polygon

        # Delegate in world order rather than scanning along imag, so this and
        # `kurven.bundle.Perimeter.contains` are the same function -- an even-odd
        # test scanning along the other axis agrees everywhere except on the
        # boundary, and the boundary is exactly where a staircase cutout lives.
        return point_in_polygon(re, im, self.corners()[:, ::-1])

    def wall_curtains(self, surface, density, *, base=0.0):
        """One occluder curtain per edge — pass straight to
        `build_occluder(..., walls=...)`. `density` is either an int (same for
        every edge) or a per-edge sequence (e.g. longer edges sampled denser)."""
        if isinstance(density, (int, np.integer)):
            density = [density] * len(self.edges)
        return [e.wall_curtain(surface, d, base=base)
                for e, d in zip(self.edges, density)]

    def to_world(self, density):
        """This perimeter as a `kurven.bundle.Perimeter` in world `(x, y)`
        order, with the per-edge sample densities the occluder used.

        The bundle's boundary description and the wall curtains that were built
        from it must agree; deriving one from the other here is what keeps them
        from being listed twice."""
        from kurven.bundle import Edge as BEdge, Perimeter as BPerimeter

        if isinstance(density, (int, np.integer)):
            density = [density] * len(self.edges)
        return BPerimeter(tuple(
            BEdge((e.start[1], e.start[0]), (e.end[1], e.end[0]), int(d))
            for e, d in zip(self.edges, density)))

    def ground_polygon_3d(self, *, z=0.0):
        """Each edge as a straight `(2, 3)` line at height `z`, in `(im, re, z)`
        — the camera-independent form of `ground_polygon`."""
        return [np.array([[e.start[0], e.start[1], z],
                          [e.end[0], e.end[1], z]]) for e in self.edges]

    def ground_polygon(self, project, *, z=0.0):
        """Each edge as a straight ground line at height `z`, projected to 2D —
        the outline the cutout casts on the base plane. Returns a list of (2, 2)
        segments (one per edge), the same corners the wall curtains rise from."""
        return [project(seg)[:, :2] for seg in self.ground_polygon_3d(z=z)]
