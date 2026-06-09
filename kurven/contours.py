from itertools import groupby

import numpy as np


def _get_paths(contour_set):
    if hasattr(contour_set, "collections") and contour_set.collections:
        return [p for c in contour_set.collections for p in c.get_paths()]
    return list(contour_set.get_paths())


def extract_contours(
    contour_set,
    height_func,
    real_bounds,
    imag_bounds,
    grid_res,
    dedup_atol=3,
):
    """Pull paths out of a matplotlib contour set, lift to 3D via `height_func`.

    `contour_set` is in grid-index coordinates (mpl convention when contouring an
    array); we rescale into real-coordinate space using `real_bounds`, `imag_bounds`,
    and `grid_res`.

    `height_func` maps a complex array (real + 1j*imag) to a real height.

    Returns (xyz, indices) where indices group points by source path.

    Dedup is O(n): endpoints are quantized to a coarse integer grid (`dedup_atol`)
    and hashed. The original `np.allclose` pairwise check was O(n^2).
    """
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    paths = _get_paths(contour_set)

    seen = set()
    kept = []
    for path in paths:
        verts = path.vertices
        if len(verts) < 1:
            continue
        key = (
            round(verts[0, 0] / dedup_atol),
            round(verts[0, 1] / dedup_atol),
            round(verts[-1, 0] / dedup_atol),
            round(verts[-1, 1] / dedup_atol),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(path)

    if not kept:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

    chunks_xy, chunks_z, chunks_i = [], [], []
    for i, path in enumerate(kept):
        xy = path.vertices.copy()
        xy[:, 0] *= (i_max - i_min) / grid_res
        xy[:, 1] *= (r_max - r_min) / grid_res
        xy[:, 1] -= (r_max - r_min) / 2
        z = height_func(xy[:, 1] + 1j * xy[:, 0])
        chunks_xy.append(xy)
        chunks_z.append(np.asarray(z).reshape(-1))
        chunks_i.append(np.full(len(xy), i, dtype=np.int64))

    xy_all = np.concatenate(chunks_xy)
    z_all = np.concatenate(chunks_z)
    xyz = np.column_stack([xy_all, z_all])
    indices = np.concatenate(chunks_i)
    return xyz, indices


def decimate_outside_critical_zone(xyz, indices, *, fine_threshold, coarse_interval):
    """Keep every point with z > fine_threshold; subsample below by `coarse_interval`."""
    if len(xyz) == 0:
        return xyz, indices
    modulus = np.mod(np.arange(len(xyz)), coarse_interval)
    mask = (xyz[:, -1] > fine_threshold) | (modulus == 0)
    return xyz[mask], indices[mask]


def mirror_x(xyz, indices, *extra_xyz_indices):
    """Mirror across the x=0 plane and concatenate. Indices in the mirror are
    offset by max(indices)+1 to avoid collisions. Extra (xyz, indices) pairs can
    be passed positionally and are concatenated after the mirror (e.g. centerline)."""
    if len(xyz) == 0:
        return xyz, indices

    mirror = xyz.copy()
    mirror[:, 0] *= -1
    offset = int(indices.max()) + 1
    mirror_idx = indices + offset

    all_xyz = [xyz, mirror]
    all_idx = [indices, mirror_idx]
    next_offset = mirror_idx.max() + 1 if len(mirror_idx) else offset
    for extra_xyz, extra_idx in extra_xyz_indices:
        if len(extra_xyz) == 0:
            continue
        all_xyz.append(extra_xyz)
        all_idx.append(extra_idx + next_offset)
        next_offset = (extra_idx + next_offset).max() + 1

    return np.vstack(all_xyz), np.hstack(all_idx)


def group_by_index(xyz, indices):
    """Yield (index, xyz_subarray) per contour group, preserving order."""
    xyzi = np.hstack([xyz, indices.reshape(-1, 1)])
    for idx, g in groupby(xyzi, lambda row: row[-1]):
        rows = np.array(list(g))
        yield int(idx), rows[:, :-1]
