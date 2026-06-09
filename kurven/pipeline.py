"""End-to-end orchestration helpers.

The full Jahnke-Emde reproduction is in `examples/gamma.py` because the
threshold/truncation logic, centerline-along-real-axis, and pole-cutoff
handling are too function-specific to belong in the library. This module
exposes a thin convenience wrapper for the generic stages:

    sample → contour → project → triangulate → rasterize → outline → clip

Use it for new functions where you don't need the gamma plate's idiosyncrasies.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

from kurven.contours import extract_contours
from kurven.outline import clip_hidden_lines, extract_outline
from kurven.projection import isometric_project
from kurven.zbuffer import ZBuffer, rasterize_triangles


def render_landscape(
    func,
    real_bounds,
    imag_bounds,
    *,
    grid_res,
    magnitude_levels,
    angle_levels,
    buffer_shape=(4000, 4000),
    x_angle_deg=-55,
    z_angle_deg=-90,
    iso_scale=0.5,
    progress=False,
):
    """Render the analytic landscape of `func` as a list of visible polylines."""
    real = np.linspace(real_bounds[0], real_bounds[1], grid_res)
    imag = np.linspace(imag_bounds[0], imag_bounds[1], grid_res)
    grid = real[:, None] + 1j * imag
    values = func(grid)
    magnitude = np.abs(values)
    angle = np.angle(values)

    fig, ax = plt.subplots()
    try:
        mag_cs = ax.contour(magnitude, levels=magnitude_levels)
        ang_cs = ax.contour(angle, levels=angle_levels)
    finally:
        plt.close(fig)

    def height(z):
        return np.abs(func(z))

    mag_xyz, mag_idx = extract_contours(mag_cs, height, real_bounds, imag_bounds, grid_res)
    ang_xyz, ang_idx = extract_contours(ang_cs, height, real_bounds, imag_bounds, grid_res)

    ang_idx_offset = ang_idx + (mag_idx.max() + 1 if len(mag_idx) else 0)
    all_xyz = np.vstack([mag_xyz, ang_xyz])
    all_idx = np.hstack([mag_idx, ang_idx_offset])

    projected = isometric_project(
        all_xyz, scale_factor=iso_scale, x_angle_deg=x_angle_deg, z_angle_deg=z_angle_deg
    )

    tri = Delaunay(projected[:, :2])
    zb = ZBuffer(
        projected[:, 0].min(),
        projected[:, 0].max(),
        projected[:, 1].min(),
        projected[:, 1].max(),
        buffer_shape,
    )
    rasterize_triangles(
        zb,
        tri.simplices,
        projected[:, 0],
        projected[:, 1],
        projected[:, 2],
        progress=progress,
    )

    segments = clip_hidden_lines(zb, projected, all_idx)
    outline = extract_outline(zb)
    return segments, outline, zb
