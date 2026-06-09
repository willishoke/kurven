from itertools import groupby

import matplotlib.pyplot as plt
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


def _paths_to_real_coords(contour_set, real_bounds, imag_bounds, grid_res, dedup_atol=3):
    """Pull paths from a mpl ContourSet, dedup, return list of (N, 2) arrays
    in real-coord space (col 0 = imag, col 1 = real). `grid_res` is either an
    int (square grid) or a (n_real, n_imag) tuple."""
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    if isinstance(grid_res, (int, np.integer)):
        n_real = n_imag = int(grid_res)
    else:
        n_real, n_imag = grid_res

    raw = _get_paths(contour_set)

    seen = set()
    kept = []
    for path in raw:
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

    out = []
    for path in kept:
        xy = path.vertices.copy()
        xy[:, 0] *= (i_max - i_min) / n_imag
        xy[:, 0] += i_min
        xy[:, 1] *= (r_max - r_min) / n_real
        xy[:, 1] += r_min
        out.append(xy)
    return out


def _split_paths_outside_zones(paths_real_coord, zones):
    """For each path (in real-coord space, col 0=imag, col 1=real), drop vertex
    runs that lie inside any zone. A path that enters and exits a zone is split
    into two paths (so a later groupby() won't bridge across the zone gap).
    """
    if not zones:
        return [p for p in paths_real_coord if len(p) >= 2]
    out = []
    for v in paths_real_coord:
        if len(v) == 0:
            continue
        keep = np.ones(len(v), dtype=bool)
        for (r_lo, r_hi, i_lo, i_hi) in zones:
            in_zone = (
                (v[:, 1] >= r_lo) & (v[:, 1] <= r_hi)
                & (v[:, 0] >= i_lo) & (v[:, 0] <= i_hi)
            )
            keep &= ~in_zone
        # Runs of consecutive True → separate output paths
        keep_int = keep.astype(np.int8)
        boundaries = np.diff(np.concatenate(([0], keep_int, [0])))
        starts = np.where(boundaries == 1)[0]
        ends = np.where(boundaries == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= 2:
                out.append(v[s:e])
    return out


def contour_adaptive(samples, levels, height_func, *, contour_op=np.abs):
    """Run mpl.contour at the coarse resolution over the full domain plus one
    pass per fine zone, drop coarse vertices inside fine zones, and lift the
    union of paths to 3D via `height_func`.

    `samples` is an `AdaptiveSamples` (from `sampling.sample_adaptive`).
    `contour_op` is applied to the complex `func()` values before contouring —
    typically `np.abs` for magnitude contours, `np.angle` for phase.

    Returns (xyz, indices), same shape conventions as `extract_contours`.
    """
    r_min, r_max = samples.real_bounds
    i_min, i_max = samples.imag_bounds

    # Coarse pass
    fig, ax = plt.subplots()
    try:
        coarse_cs = ax.contour(contour_op(samples.coarse_values), levels=levels)
    finally:
        plt.close(fig)

    coarse_paths = _paths_to_real_coords(
        coarse_cs, samples.real_bounds, samples.imag_bounds, samples.coarse_res
    )
    coarse_paths = _split_paths_outside_zones(coarse_paths, samples.fine_zones)

    all_paths = list(coarse_paths)

    # Per-zone fine passes
    for zvalues, zone, zres in zip(
        samples.fine_values, samples.fine_zones, samples.fine_resolutions
    ):
        zr_min, zr_max, zi_min, zi_max = zone
        fig, ax = plt.subplots()
        try:
            fine_cs = ax.contour(contour_op(zvalues), levels=levels)
        finally:
            plt.close(fig)
        fine_paths = _paths_to_real_coords(
            fine_cs, (zr_min, zr_max), (zi_min, zi_max), grid_res=zres
        )
        all_paths.extend(p for p in fine_paths if len(p) >= 2)

    if not all_paths:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

    chunks_xyz, chunks_idx = [], []
    for i, xy in enumerate(all_paths):
        z = height_func(xy[:, 1] + 1j * xy[:, 0])
        chunks_xyz.append(np.column_stack([xy, np.asarray(z).reshape(-1)]))
        chunks_idx.append(np.full(len(xy), i, dtype=np.int64))

    return np.concatenate(chunks_xyz), np.concatenate(chunks_idx)
