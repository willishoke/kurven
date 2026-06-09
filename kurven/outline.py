from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np


def extract_outline(zb, *, cutoff_min=None, cutoff_max=None):
    """Trace the silhouette of the filled region of a Z-buffer.

    Binarize (filled→1, empty→-1), find the level-0 contour in pixel space,
    map back to coord space. Optional axis-aligned clip via cutoff_min/max
    (each a length-2 array; semantics match the original notebook's
    `outline_min_cutoff` / `outline_max_cutoff`).
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
        paths = [p for c in cs.collections for p in c.get_paths()]
    else:
        paths = list(cs.get_paths())
    if not paths:
        return np.zeros((0, 2))

    pts = zb.index_to_coord(paths[0].vertices)

    mask = np.ones(len(pts), dtype=bool)
    if cutoff_min is not None:
        mask &= np.all(pts > np.asarray(cutoff_min), axis=1)
    if cutoff_max is not None:
        mask &= np.all(pts < np.asarray(cutoff_max), axis=1)
    return pts[mask]


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
