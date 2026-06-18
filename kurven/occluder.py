"""Occluder mesh assembly: a (optionally tiled) heightfield plus cut-face wall
curtains, concatenated into one mesh for the Z-buffer.

The occluder is the set of surfaces that actually block sight lines for
hidden-line removal. A `Surface` already triangulates its own clamped
heightfield (`Surface.grid_mesh`); this module tiles that base for periodic
landscapes, adds the vertical cut-face curtains, and concatenates everything
with correctly offset triangle indices. A periodic landscape (tiled, with cut
walls) and a single-tile landscape both build their occluder through
`build_occluder`, passing `tile_transforms` or not.
"""

import numpy as np


def wall_curtain(im, re, surface, *, base=0.0):
    """A vertical ruled strip along a perimeter polyline, from z=`base` up to
    the surface height.

    `im`, `re` are equal-length 1D arrays giving the perimeter in domain
    coordinates. Returns `(vertices, triangles)` where vertices are (im, re, z).
    """
    im = np.asarray(im, dtype=float)
    re = np.asarray(re, dtype=float)
    h = surface.height_at(re, im)
    perim = np.column_stack([im, re])
    n = len(perim)
    a = np.arange(n - 1)
    vertices = np.vstack([np.column_stack([perim, h]),
                          np.column_stack([perim, np.full(n, base)])])
    triangles = np.vstack([np.stack([a, a + 1, a + n], axis=1),
                           np.stack([a + 1, a + 1 + n, a + n], axis=1)])
    return vertices, triangles


def concat_meshes(meshes):
    """Concatenate `(vertices, triangles)` meshes, offsetting triangle indices
    into the combined vertex array. Returns `(vertices, triangles)`."""
    vlist, tlist, offset = [], [], 0
    for vertices, triangles in meshes:
        vlist.append(vertices)
        tlist.append(np.asarray(triangles) + offset)
        offset += len(vertices)
    if not vlist:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(vlist), np.vstack(tlist)


def build_occluder(surface, step=1, *, walls=(), tile_transforms=None):
    """Build a Z-buffer occluder mesh from a surface, optional tiling, and walls.

    surface:         a `Surface`; its clamped heightfield is meshed via
                     `grid_mesh(step)`.
    step:            occluder subsample step.
    tile_transforms: optional iterable of callables `V -> V` applied to the base
                     grid vertices to tile/reflect a periodic landscape. The
                     untransformed base tile is always included.
    walls:           iterable of `(vertices, triangles)` wall-curtain meshes
                     (see `wall_curtain`).

    Returns `(vertices, triangles)`.
    """
    base_v, base_t = surface.grid_mesh(step)
    pieces = [(base_v, base_t)]
    if tile_transforms:
        pieces.extend((xform(base_v), base_t) for xform in tile_transforms)
    pieces.extend(walls)
    return concat_meshes(pieces)
