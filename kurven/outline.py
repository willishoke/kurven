from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


def _split_path_by_moves(path):
    verts = path.vertices
    codes = path.codes
    if codes is None or len(verts) == 0:
        return [verts] if len(verts) else []
    moves = np.where(codes == MplPath.MOVETO)[0]
    if len(moves) <= 1:
        return [verts]
    boundaries = list(moves) + [len(verts)]
    return [verts[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]


def extract_outline(zb, *, cutoff_min=None, cutoff_max=None):
    """Trace the silhouette of the filled region of a Z-buffer.

    Binarize (filled→1, empty→-1), find the level-0 contour in pixel space,
    map back to coord space. Returns ALL silhouette polylines joined with NaN
    separators, so `plt.plot(*outline.T)` renders disconnected components as
    separate strokes rather than bridging them with straight lines.
    """
    binarized = zb.buffer.T.copy()
    binarized[np.isnan(binarized)] = -1
    binarized[np.isinf(binarized)] = -1
    binarized[binarized > 0] = 1
    binarized[binarized < 0] = -1

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(binarized, [0])
    finally:
        plt.close(fig)

    if hasattr(cs, "collections") and cs.collections:
        raw_paths = [p for c in cs.collections for p in c.get_paths()]
    else:
        raw_paths = list(cs.get_paths())
    if not raw_paths:
        return np.zeros((0, 2))

    polylines = []
    for p in raw_paths:
        polylines.extend(_split_path_by_moves(p))

    sep = np.array([[np.nan, np.nan]])
    pieces = []
    for verts in polylines:
        if len(verts) < 2:
            continue
        pts = zb.index_to_coord(verts)
        if cutoff_min is not None or cutoff_max is not None:
            mask = np.ones(len(pts), dtype=bool)
            if cutoff_min is not None:
                mask &= np.all(pts > np.asarray(cutoff_min), axis=1)
            if cutoff_max is not None:
                mask &= np.all(pts < np.asarray(cutoff_max), axis=1)
            # Drop full-run if no pixels survive; otherwise keep with internal NaNs
            # at mask transitions so plot doesn't bridge across gaps.
            if not mask.any():
                continue
            # Split into runs of True
            keep_int = mask.astype(np.int8)
            boundaries = np.diff(np.concatenate(([0], keep_int, [0])))
            starts = np.where(boundaries == 1)[0]
            ends = np.where(boundaries == -1)[0]
            for s, e in zip(starts, ends):
                if e - s >= 2:
                    pieces.append(pts[s:e])
                    pieces.append(sep)
        else:
            pieces.append(pts)
            pieces.append(sep)

    if not pieces:
        return np.zeros((0, 2))
    # Trailing sep is fine for plot but drop it for cleanliness
    if np.all(np.isnan(pieces[-1])):
        pieces = pieces[:-1]
    return np.vstack(pieces)


def clip_hidden_lines(zb, xyz_view, indices, *, margin=0.01):
    """Split each contour group into visible line segments using the Z-buffer.

    A point is visible if `xyz_view[i, 2] + margin > zb.buffer[pixel(i)]`.
    Hidden runs split a contour into multiple visible segments; segments of
    length < 2 are dropped.

    Returns: list of (M, 2) arrays — each one a plottable polyline in view-space.
    """
    n = len(indices)
    if n == 0:
        return []

    pixel = zb.coord_to_index(xyz_view[:, :2]).astype(np.int64)
    bh, bw = zb.buffer.shape

    in_buffer = (
        (pixel[:, 0] >= 0)
        & (pixel[:, 0] < bh)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < bw)
    )
    buffer_z = np.full(n, np.inf)
    buffer_z[in_buffer] = zb.buffer[pixel[in_buffer, 0], pixel[in_buffer, 1]]
    visible = (xyz_view[:, 2] + margin) > buffer_z

    segments = []
    for _, group_iter in groupby(range(n), lambda i: indices[i]):
        group = list(group_iter)
        current = []
        for i in group:
            if visible[i]:
                current.append(xyz_view[i, :2])
            else:
                if len(current) >= 2:
                    segments.append(np.array(current))
                current = []
        if len(current) >= 2:
            segments.append(np.array(current))
    return segments
