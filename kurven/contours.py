from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


def _get_paths(contour_set):
    if hasattr(contour_set, "collections") and contour_set.collections:
        return [p for c in contour_set.collections for p in c.get_paths()]
    return list(contour_set.get_paths())


def _split_path_by_moves(path):
    """Split a `matplotlib.path.Path` into a list of (N, 2) vertex arrays, one
    per contiguous sub-polyline (a `MOVETO` code starts a new sub-polyline).

    Without this, a single Path representing several disconnected contour loops
    (e.g. one loop per pole) gets flattened to a single (N, 2) array, and any
    code that just plots `path.vertices` bridges across the disjoint loops.
    """
    verts = path.vertices
    codes = path.codes
    if codes is None or len(verts) == 0:
        return [verts] if len(verts) else []
    moves = np.where(codes == MplPath.MOVETO)[0]
    if len(moves) <= 1:
        return [verts]
    boundaries = list(moves) + [len(verts)]
    out = []
    for i in range(len(boundaries) - 1):
        seg = verts[boundaries[i]:boundaries[i + 1]]
        if len(seg) >= 2:
            out.append(seg)
    return out


def _get_subpolylines(contour_set):
    """Yield (vertices_in_grid_index_space,) for every distinct sub-polyline
    across every path in the contour set."""
    out = []
    for path in _get_paths(contour_set):
        out.extend(_split_path_by_moves(path))
    return out


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
    polylines = _get_subpolylines(contour_set)

    seen = set()
    kept = []
    for verts in polylines:
        if len(verts) < 2:
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
        kept.append(verts)

    if not kept:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

    chunks_xy, chunks_z, chunks_i = [], [], []
    for i, verts in enumerate(kept):
        xy = verts.copy()
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
    """Pull paths from a mpl ContourSet, split each into sub-polylines on
    MOVETO codes, dedup, and convert vertices to real-coord space
    (col 0 = imag, col 1 = real). `grid_res` is either an int (square grid)
    or a (n_real, n_imag) tuple."""
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    if isinstance(grid_res, (int, np.integer)):
        n_real = n_imag = int(grid_res)
    else:
        n_real, n_imag = grid_res

    polylines = _get_subpolylines(contour_set)

    seen = set()
    kept = []
    for verts in polylines:
        if len(verts) < 2:
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
        kept.append(verts)

    out = []
    for verts in kept:
        xy = verts.copy()
        xy[:, 0] *= (i_max - i_min) / n_imag
        xy[:, 0] += i_min
        xy[:, 1] *= (r_max - r_min) / n_real
        xy[:, 1] += r_min
        out.append(xy)
    return out


def _segs_per_level_to_real_coords(
    contour_set, levels, real_bounds, imag_bounds, grid_res, dedup_atol=3,
):
    """Use `contour_set.allsegs` to organize contour segments by level.

    Returns dict {level_value: [list of (N, 2) arrays in real-coord space]}.
    `allsegs` already separates disconnected pieces per level, so we don't need
    the `MOVETO`-splitting fallback. Dedup is per-level.
    """
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    if isinstance(grid_res, (int, np.integer)):
        n_real = n_imag = int(grid_res)
    else:
        n_real, n_imag = grid_res

    if not hasattr(contour_set, "allsegs"):
        return {}

    # `seen` spans ALL levels: matching the original notebook's extractContours,
    # which deduped paths globally across a single ContourSet. Without this, when
    # mpl emits the same physical curve at two levels (e.g. angle=+π and angle=−π
    # tracing the same branch-cut crossing), both copies survive and stack as a
    # bundle of overlapping wavy lines in the final render.
    seen = set()
    out = {}
    for level_idx, segs in enumerate(contour_set.allsegs):
        level = float(levels[level_idx])
        kept = []
        for seg in segs:
            if len(seg) < 2:
                continue
            key = (
                round(seg[0, 0] / dedup_atol),
                round(seg[0, 1] / dedup_atol),
                round(seg[-1, 0] / dedup_atol),
                round(seg[-1, 1] / dedup_atol),
            )
            if key in seen:
                continue
            seen.add(key)
            xy = seg.copy()
            xy[:, 0] *= (i_max - i_min) / n_imag
            xy[:, 0] += i_min
            xy[:, 1] *= (r_max - r_min) / n_real
            xy[:, 1] += r_min
            kept.append(xy)
        out[level] = kept
    return out


def _near_zone_boundary(pt, zones, tolerance):
    """True if `pt` (in real-coord space, col 0=imag, col 1=real) is within
    `tolerance` of any zone boundary edge."""
    x, y = pt[0], pt[1]
    for r_lo, r_hi, i_lo, i_hi in zones:
        if (i_lo - tolerance <= x <= i_hi + tolerance
                and r_lo - tolerance <= y <= r_hi + tolerance):
            dist_to_edge = min(
                abs(x - i_lo), abs(x - i_hi),
                abs(y - r_lo), abs(y - r_hi),
            )
            if dist_to_edge <= tolerance:
                return True
    return False


def _stitch_paths(paths, zones, tolerance):
    """Greedy endpoint matching: pair distinct paths whose endpoints are within
    `tolerance` and at least one of which is near a zone boundary. Concatenate
    matched paths into single polylines (reversing as needed).

    Avoids welding unrelated nearby contours in dense regions by requiring
    proximity to a zone boundary, where coarse/fine seams actually live.
    """
    if len(paths) < 2 or not zones:
        return list(paths)

    from scipy.spatial import cKDTree

    n = len(paths)
    positions = np.empty((2 * n, 2))
    near_boundary = np.zeros(2 * n, dtype=bool)
    for i, p in enumerate(paths):
        positions[2 * i] = p[0]
        positions[2 * i + 1] = p[-1]
        near_boundary[2 * i] = _near_zone_boundary(p[0], zones, tolerance)
        near_boundary[2 * i + 1] = _near_zone_boundary(p[-1], zones, tolerance)

    tree = cKDTree(positions)
    pairs = tree.query_pairs(tolerance, output_type="ndarray")
    if len(pairs) == 0:
        return list(paths)

    dists = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    order = np.argsort(dists)

    partner = np.full(2 * n, -1, dtype=np.int64)
    for k in order:
        a, b = int(pairs[k, 0]), int(pairs[k, 1])
        if a // 2 == b // 2:
            continue
        if partner[a] != -1 or partner[b] != -1:
            continue
        if not (near_boundary[a] or near_boundary[b]):
            continue
        partner[a] = b
        partner[b] = a

    visited = np.zeros(n, dtype=bool)
    out = []
    for start in range(n):
        if visited[start]:
            continue
        if partner[2 * start] == -1:
            free_label = 0
        elif partner[2 * start + 1] == -1:
            free_label = 1
        else:
            free_label = 0  # cycle — start anywhere

        chain = []
        seg = paths[start]
        if free_label == 1:
            seg = seg[::-1]
        chain.append(seg)
        visited[start] = True
        current_idx = start
        tail_label = 1 - free_label

        while True:
            tail_endpoint = 2 * current_idx + tail_label
            next_endpoint = partner[tail_endpoint]
            if next_endpoint == -1:
                break
            next_path_idx = next_endpoint // 2
            next_label = next_endpoint % 2
            if visited[next_path_idx]:
                break

            seg = paths[next_path_idx]
            if next_label == 1:
                seg = seg[::-1]
            chain.append(seg)
            visited[next_path_idx] = True
            current_idx = next_path_idx
            tail_label = 1 - next_label

        if len(chain) == 1:
            out.append(chain[0])
        else:
            parts = [chain[0]]
            for seg in chain[1:]:
                if len(seg) >= 2:
                    parts.append(seg[1:])
            out.append(np.concatenate(parts))

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


def contour_adaptive(samples, levels, height_func, *, contour_op=np.abs, stitch_tolerance=None):
    """Run mpl.contour at the coarse resolution over the full domain plus one
    pass per fine zone, drop coarse vertices inside fine zones, optionally
    stitch matching coarse/fine paths at zone boundaries, and lift the union
    to 3D via `height_func`.

    `samples` is an `AdaptiveSamples` (from `sampling.sample_adaptive`).
    `contour_op` is applied to the complex `func()` values before contouring —
    typically `np.abs` for magnitude contours, `np.angle` for phase.

    `stitch_tolerance`: max distance (in real-coord space) for welding two
    same-level path endpoints. Use a value somewhat smaller than the local
    contour spacing; ~1× a coarse cell is a good default. None disables.

    Returns (xyz, indices), same shape conventions as `extract_contours`.
    """
    levels_arr = np.asarray(levels, dtype=float)

    fig, ax = plt.subplots()
    try:
        coarse_cs = ax.contour(contour_op(samples.coarse_values), levels=levels)
    finally:
        plt.close(fig)
    coarse_by_level = _segs_per_level_to_real_coords(
        coarse_cs, levels_arr,
        samples.real_bounds, samples.imag_bounds, samples.coarse_res,
    )
    coarse_by_level = {
        lvl: _split_paths_outside_zones(segs, samples.fine_zones)
        for lvl, segs in coarse_by_level.items()
    }

    fine_by_level = {}
    for zvalues, zone, zres in zip(
        samples.fine_values, samples.fine_zones, samples.fine_resolutions
    ):
        zr_min, zr_max, zi_min, zi_max = zone
        fig, ax = plt.subplots()
        try:
            fine_cs = ax.contour(contour_op(zvalues), levels=levels)
        finally:
            plt.close(fig)
        zone_by_level = _segs_per_level_to_real_coords(
            fine_cs, levels_arr, (zr_min, zr_max), (zi_min, zi_max), zres,
        )
        for lvl, segs in zone_by_level.items():
            fine_by_level.setdefault(lvl, []).extend(segs)

    all_levels = sorted(set(coarse_by_level) | set(fine_by_level))
    chunks_xyz, chunks_idx = [], []
    path_idx = 0
    for lvl in all_levels:
        combined = coarse_by_level.get(lvl, []) + fine_by_level.get(lvl, [])
        if stitch_tolerance is not None and samples.fine_zones and len(combined) >= 2:
            combined = _stitch_paths(combined, samples.fine_zones, stitch_tolerance)
        for xy in combined:
            if len(xy) < 2:
                continue
            z = height_func(xy[:, 1] + 1j * xy[:, 0])
            chunks_xyz.append(np.column_stack([xy, np.asarray(z).reshape(-1)]))
            chunks_idx.append(np.full(len(xy), path_idx, dtype=np.int64))
            path_idx += 1

    if not chunks_xyz:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks_xyz), np.concatenate(chunks_idx)
