"""Hatching: the ink that shades a cut face and a truncated top.

The plates shade two kinds of flat surface. A **cut face** -- the vertical wall
where the landscape is sliced open -- is drawn with vertical strokes from the
ground up to the crest. A **truncated top** -- the plateau where a pole spire
was cut off -- is drawn with parallel strokes at the cap, bounded by the rim
where |f| drops back under it. Every published plate in `examples/` writes both
by hand, with hand-found spire radii and `while` loops that step outward until
the magnitude falls below the limit; this module is the general form of all of
it, derived from the grids and the cap rather than from numbers typed per plate.

That generality is the point: a landscape whose function, domain and cap the
user is choosing cannot have its hatching written down in advance. The four
`LayerSource` kinds in `kurven.bundle` (`wallHatch`, `wallOutline`, `capHatch`,
`capOutline`) *describe* the hatching, and this is the reference implementation
they are defined by. The Swift reader derives the same ink from the same
description, and `tests/fixtures/hatch` is where the two are held to each other.

Everything here works in **world** order `(x = real, y = imag, z)`, like the
bundle and unlike the rest of the library, because a description in a manifest
is what it consumes and what it produces is destined for one.

Where the heights come from
---------------------------
A stroke's top is the capped surface, and there are two defensible ways to ask
for it: evaluate f exactly, or interpolate the sampled grid. They differ by the
grid's interpolation error, and the two sides of the contract necessarily
differ -- Python has the function and Swift has only the grid. So the height is a
*parameter* here (`height`, `magnitude`), and the caller says which it means:
the plate passes its analytic `Surface`, the fixtures pass `grid_sampler`, which
is `Grid2D.sample` transcribed. Anything that solves for a crossing -- the cap
strokes and the rim -- reads the grid either way, because a crossing between two
samples is a statement about the samples.
"""

from __future__ import annotations

import numpy as np

from kurven.bundle import (LayerCapHatch, LayerCapOutline, LayerWallHatch,
                           LayerWallOutline, RealBandCaps)


# --------------------------------------------------------------------------
# reading the grid the way the consumer does
# --------------------------------------------------------------------------


def grid_positions(domain, n, axis):
    """Where the grid's samples sit along one axis.

    `Grid2D.position`, transcribed including its arithmetic order: the two
    implementations have to agree about sample coordinates to the last bit, or
    every crossing they solve lands somewhere slightly different.
    """
    interval = domain.real if axis == "real" else domain.imag
    length = interval.hi - interval.lo
    return interval.lo + length * np.arange(n, dtype=float) / max(n - 1, 1)


def grid_sampler(grid, domain, *, nearest=False):
    """`(x, y) -> value`, bilinear (or nearest), clamped at the edges.

    `Grid2D.sample` and `Grid2D.nearest`, transcribed. `grid` is `(ny, nx)` with
    rows indexing imag, as `height.npy` is written.
    """
    ny, nx = grid.shape
    values = np.asarray(grid, dtype=np.float64)

    def index(x, y):
        fx = ((np.asarray(x, dtype=float) - domain.real.lo)
              / max(domain.real.hi - domain.real.lo, np.finfo(float).tiny)) * (nx - 1)
        fy = ((np.asarray(y, dtype=float) - domain.imag.lo)
              / max(domain.imag.hi - domain.imag.lo, np.finfo(float).tiny)) * (ny - 1)
        return fx, fy

    def sample(x, y):
        fx, fy = index(x, y)
        if nearest:
            ix = np.clip(np.rint(fx).astype(np.int64), 0, nx - 1)
            iy = np.clip(np.rint(fy).astype(np.int64), 0, ny - 1)
            return values[iy, ix]
        x0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 1)
        y0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 1)
        x1 = np.minimum(x0 + 1, nx - 1)
        y1 = np.minimum(y0 + 1, ny - 1)
        tx = np.clip(fx - x0, 0.0, 1.0)
        ty = np.clip(fy - y0, 0.0, 1.0)
        a = values[y0, x0] * (1 - tx) + values[y0, x1] * tx
        b = values[y1, x0] * (1 - tx) + values[y1, x1] * tx
        return a * (1 - ty) + b * ty

    return sample


# --------------------------------------------------------------------------
# cut faces
# --------------------------------------------------------------------------


def edge_points(edge, n):
    """`n` points from an edge's start to its end, `Mesh.wallCurtain`'s
    parameterization: `p = a(1 - t) + bt`, which is not `a + (b - a)t` in
    floating point and has to be the one the curtain used, or the hatch stands a
    rounding error off the wall it hatches."""
    t = np.arange(n, dtype=float) / max(n - 1, 1)
    x = edge.start[0] * (1 - t) + edge.end[0] * t
    y = edge.start[1] * (1 - t) + edge.end[1] * t
    return x, y


def _stroke_count(length, spacing):
    """Samples along an edge of this length, hatched at this spacing.

    `floor(L / spacing + 0.5)` rather than a language's own rounding, because
    Python rounds halves to even and Swift rounds them away from zero, and a
    landscape whose edge happens to be an exact multiple of the spacing would
    otherwise be hatched differently by the two.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        return 2
    return max(2, int(np.floor(length / spacing + 0.5)) + 1)


def _vertical(x, y, base, top, pitch):
    """One vertical stroke as a polyline, subdivided every `pitch`.

    Not a two-point segment: the bake clips per vertex, so a stroke whose middle
    is behind a ridge and whose ends are not would be drawn straight through it.
    """
    if not (top > base):
        return None
    steps = 1
    if np.isfinite(pitch) and pitch > 0:
        steps = max(1, int(np.ceil((top - base) / pitch)))
    z = base + (top - base) * np.arange(steps + 1, dtype=float) / steps
    return np.column_stack([np.full(steps + 1, x), np.full(steps + 1, y), z])


def wall_hatch(source, perimeter, height):
    """`LayerWallHatch` -> a list of `(N, 3)` world polylines.

    `perimeter` is a `kurven.bundle.Perimeter` (world order) and `height(x, y)`
    gives the capped surface height.
    """
    out = []
    for index in source.edges:
        if index < 0 or index >= len(perimeter.edges):
            continue
        edge = perimeter.edges[index]
        length = float(np.hypot(edge.end[0] - edge.start[0],
                                edge.end[1] - edge.start[1]))
        n = _stroke_count(length, source.spacing)
        x, y = edge_points(edge, n)
        if source.trim:
            x, y = x[1:-1], y[1:-1]
        tops = np.asarray(height(x, y), dtype=float) + source.top_offset
        for xi, yi, top in zip(x, y, tops):
            stroke = _vertical(xi, yi, source.base, float(top), source.pitch)
            if stroke is not None:
                out.append(stroke)
    return out


def wall_outline(source, perimeter, height):
    """`LayerWallOutline` -> crest and foot per edge, then a post per corner.

    The corners are deduplicated by exact coordinate, so a closed traversal
    stands one post at each of its corners rather than two.
    """
    out = []
    corners = []
    for index in source.edges:
        if index < 0 or index >= len(perimeter.edges):
            continue
        edge = perimeter.edges[index]
        x, y = edge_points(edge, max(int(edge.density), 2))
        z = np.asarray(height(x, y), dtype=float)
        out.append(np.column_stack([x, y, z]))
        out.append(np.column_stack([x, y, np.full(len(x), source.base)]))
        for corner in (edge.start, edge.end):
            if corner not in corners:
                corners.append(corner)
    for cx, cy in corners:
        top = float(np.asarray(height(np.array([cx]), np.array([cy])))[0])
        post = _vertical(cx, cy, source.base, top, source.pitch)
        if post is not None:
            out.append(post)
    return out


# --------------------------------------------------------------------------
# truncated tops
# --------------------------------------------------------------------------


def cap_breakpoints(caps, lo, hi):
    """Where the cap changes across `[lo, hi]`, as the interval ends it makes.

    `RealBandCaps` is a staircase in Re tested in order, so the values that
    matter are its thresholds, sorted and clipped to the span. Everything else
    is one interval. Returned as consecutive `(a, b)` pairs covering `[lo, hi]`;
    the cap is constant on `[a, b)`.
    """
    edges = [lo]
    if isinstance(caps, RealBandCaps):
        for band in caps.bands:
            if lo < band.below < hi:
                edges.append(float(band.below))
    edges = sorted(set(edges)) + [hi]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def _runs_at_or_above(coord, excess):
    """The stretches where `excess >= 0`, with ends solved linearly.

    Returns a list of `(vertices, )` coordinate arrays along the line: the
    samples inside each run, extended to the sign change on either side. The
    crossing is `t = e0 / (e0 - e1)`, which is where the same linear
    interpolation would put the rim contour, so a stroke ends on the rim rather
    than at the last sample inside it.
    """
    inside = excess >= 0
    if not inside.any():
        return []
    runs = []
    edges = np.diff(np.concatenate([[0], inside.view(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        pieces = [coord[a:b]]
        if a > 0:
            e0, e1 = excess[a - 1], excess[a]
            t = e0 / (e0 - e1) if e0 != e1 else 0.0
            pieces.insert(0, [coord[a - 1] + t * (coord[a] - coord[a - 1])])
        if b < len(coord):
            e0, e1 = excess[b - 1], excess[b]
            t = e0 / (e0 - e1) if e0 != e1 else 0.0
            pieces.append([coord[b - 1] + t * (coord[b] - coord[b - 1])])
        line = np.concatenate([np.asarray(p, dtype=float) for p in pieces])
        if len(line) >= 2:
            runs.append(line)
    return runs


def cap_hatch(source, domain, shape, caps, magnitude):
    """`LayerCapHatch` -> a list of `(N, 3)` world polylines at the cap.

    `shape` is the height grid's `(ny, nx)`; the strokes are solved on the same
    columns (or rows) the grid has, so they end where the rim is.
    """
    ny, nx = shape
    along = source.axis
    across_interval = domain.imag if source.axis == "real" else domain.real
    along_interval = domain.real if source.axis == "real" else domain.imag
    samples = grid_positions(domain, nx if source.axis == "real" else ny, along)

    out = []
    spacing = float(source.spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        return out
    # Anchored to whole multiples of the spacing rather than to the domain, so
    # the ruling stays put when the domain moves under it.
    first = int(np.ceil(across_interval.lo / spacing))
    last = int(np.floor(across_interval.hi / spacing))
    for k in range(first, last + 1):
        c = k * spacing
        if not (across_interval.lo < c < across_interval.hi):
            continue
        # The cap varies with Re and with nothing else. A line along the real
        # axis therefore crosses the bands and is solved piece by piece; a line
        # along the imaginary axis stays in one band for its whole length, and
        # splitting it at thresholds measured on the other axis would be a
        # coordinate confusion rather than a subdivision.
        if source.axis == "real":
            intervals = cap_breakpoints(caps, along_interval.lo, along_interval.hi)
        else:
            intervals = [(along_interval.lo, along_interval.hi)]
        for a, b in intervals:
            coord = samples[(samples > a) & (samples < b)]
            coord = np.concatenate([[a], coord, [b]])
            x, y = (coord, np.full(len(coord), c)) if source.axis == "real" \
                else (np.full(len(coord), c), coord)
            # The cap is constant on [a, b); read it just inside, so a band's
            # own threshold does not pick up the next band's cap.
            cap = float(np.asarray(caps.at(np.array([a if source.axis == "real"
                                                     else c])))[0])
            if not np.isfinite(cap):
                continue
            excess = np.asarray(magnitude(x, y), dtype=float) - cap
            for line in _runs_at_or_above(coord, excess):
                lx, ly = (line, np.full(len(line), c)) if source.axis == "real" \
                    else (np.full(len(line), c), line)
                out.append(np.column_stack([lx, ly, np.full(len(line), cap)]))
    return out


def cap_excess(height_grid, domain, caps):
    """|f| - cap, as a grid, in float32.

    The rim is the zero contour of this. float32 because that is what
    `height.npy` holds and what the Swift side subtracts in; a float64
    intermediate here would put the contour in a slightly different place than
    the consumer does, for no gain -- the samples themselves are float32.
    """
    ny, nx = height_grid.shape
    x = grid_positions(domain, nx, "real")
    cap = np.asarray(caps.at(x), dtype=np.float32)
    return np.asarray(height_grid, dtype=np.float32) - cap[None, :]


def lift_to_cap(xy, caps):
    """Put `(N, 2)` world points at the cap above them."""
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([xy, np.asarray(caps.at(xy[:, 0]), dtype=float)])


def cap_outline(source, height_grid, domain, caps, *, chunk_count=1):
    """`LayerCapOutline` -> the rim of every truncated top, in world order.

    contourpy is the oracle for marching squares in this codebase, so the rim is
    one of its contours rather than a second implementation of the same walk:
    the zero level of `cap_excess`. `contour_levels` wants the library's
    `(n_real, n_imag)` array and hands back `(imag, real)` columns, so this is
    also where that exchange happens for the rim.
    """
    from kurven.contours import contour_levels

    excess = cap_excess(height_grid, domain, caps)
    if not np.isfinite(excess).any():
        return []
    rb = (domain.real.lo, domain.real.hi)
    ib = (domain.imag.lo, domain.imag.hi)
    out = []
    for _, paths in contour_levels(excess.T, [0.0], rb, ib, chunk_count=chunk_count):
        for path in paths:
            out.append(lift_to_cap(np.column_stack([path[:, 1], path[:, 0]]), caps))
    return out


# --------------------------------------------------------------------------
# the dispatcher
# --------------------------------------------------------------------------


def clip_to_region(paths, region):
    """Split each path at the region boundary, keeping the runs inside.

    The same rule `Surface.lift_contours` applies to contours, and for the same
    reason: filtering vertices and keeping the survivors as one polyline welds
    the far side of a stroke to the near side across ground the region has
    deliberately removed.
    """
    from kurven.bundle import InsideRegion

    if not isinstance(region, InsideRegion):
        return list(paths)
    out = []
    for path in paths:
        mask = np.asarray(region.perimeter.contains(path[:, 0], path[:, 1]), dtype=bool)
        if mask.all():
            out.append(path)
            continue
        edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= 2:
                out.append(path[a:b])
    return out


def replicate(paths, tiles):
    """One copy per occluder tile, the maps the heightfield is instanced by."""
    return [tile.apply(path) for tile in tiles for path in paths]


def derive(source, *, perimeter=None, region=None, tiles=(), height=None,
           magnitude=None, height_grid=None, domain=None, caps=None,
           chunk_count=1):
    """Any of the four hatching sources -> world-order polylines.

    Returns `None` for a source this module does not own, so a caller can use it
    as the "is this hatching?" test as well as the derivation.
    """
    if isinstance(source, LayerWallHatch):
        if perimeter is None:
            return []
        paths = wall_hatch(source, perimeter, height)
    elif isinstance(source, LayerWallOutline):
        if perimeter is None:
            return []
        paths = wall_outline(source, perimeter, height)
    elif isinstance(source, LayerCapHatch):
        paths = cap_hatch(source, domain, height_grid.shape, caps, magnitude)
        if source.tiled:
            paths = replicate(paths, tiles)
    elif isinstance(source, LayerCapOutline):
        paths = cap_outline(source, height_grid, domain, caps,
                            chunk_count=chunk_count)
        if source.tiled:
            paths = replicate(paths, tiles)
    else:
        return None
    return clip_to_region(paths, region) if region is not None else paths
