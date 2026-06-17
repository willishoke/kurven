"""Riemann zeta analytic landscape.

Direct port of 2401/zeta.ipynb. Uses kurven for the rasterizer and clip
primitives but follows the notebook's Z-buffer construction faithfully:
Delaunay-triangulate the union of contour vertices + boundary curves +
peak contours, rasterize that mesh, clip the major contours against it.

The notebook's approach naturally produces the right cutout because the
contour vertices live only inside the kept region (after the c1|c2|c3|c4
filter), and the triangulation only covers regions with vertices — so
the buffer has a low/empty region exactly where the staircase cutout is.

Run:
    python examples/zeta.py --gpu
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R

from kurven.bench import PhaseTimer
from kurven.contours import contour_levels
from kurven.outline import clip_hidden_lines
from kurven.scaffold import wall_hatch
from kurven.surface import Surface
from kurven.zbuffer import (
    ZBuffer,
    rasterize_triangles,
    rasterize_triangles_gpu,
)


DEFAULT_CACHE = "/Users/willishoke/journal/2401/zeta_5000.npy"


def _cutout_kept(xyz_view):
    """Mirror of cell 9's c1|c2|c3|c4 filter — keep contour vertices inside
    the staircase kept region. xyz_view col 0=imag, col 1=real."""
    im = xyz_view[:, 0]
    re = xyz_view[:, 1]
    c1 = (im > 0) & (im < 28)
    c2 = (im > -5) & (re > -2) & (im < 28) & (im > -28)
    c3 = (im > -15) & (re > -0.5) & (im < 28) & (im > -28)
    c4 = (re > 0.5) & (im > -28) & (im < 28)
    return c1 | c2 | c3 | c4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE)
    parser.add_argument("--buffer", type=int, default=8000)
    parser.add_argument("--output-prefix", type=str, default="zeta")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--clip-margin", type=float, default=0.2,
                        help="Matches the notebook's cell 22 margin.")
    args = parser.parse_args()

    if args.backend:
        matplotlib.use(args.backend)
    buffer_shape = (args.buffer, args.buffer)
    progress = not args.no_progress
    timer = PhaseTimer()

    r_min, r_max = -6.0, 8.0
    i_min, i_max = -30.0, 30.0
    z_limit = 6.0

    timer.tick("load cache")
    comp = np.load(args.cache)
    surface = Surface.from_cache(comp, (r_min, r_max), (i_min, i_max), z_limit=z_limit)
    mag, angle = surface.mag, surface.angle
    print(f"      cached grid: {comp.shape}, |zeta| ∈ "
          f"[{mag.min():.3f}, {mag.max():.3f}]")

    # Contour levels exactly per the notebook
    mag_major_levels = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    angle_major_levels = np.linspace(-np.pi, np.pi, 11)
    peak_levels = [2.5, 3.5, 4.5, 5.5]

    timer.tick("contour generation")
    mag_paths = contour_levels(mag, mag_major_levels, (r_min, r_max), (i_min, i_max))
    angle_paths = contour_levels(angle, angle_major_levels, (r_min, r_max), (i_min, i_max))
    peak_paths = contour_levels(mag, peak_levels, (r_min, r_max), (i_min, i_max))

    timer.tick("assemble major_data")
    mag_chunks_xyz, mag_chunks_idx = [], []
    path_idx = 0
    for lvl, paths in mag_paths:
        for xy in paths:
            z = np.full(len(xy), lvl)
            xyz = np.column_stack([xy, z])
            # Apply cutout filter (matches cell 9's c1|c2|c3|c4)
            keep = _cutout_kept(xyz)
            xyz = xyz[keep]
            if len(xyz) < 2:
                continue
            mag_chunks_xyz.append(xyz)
            mag_chunks_idx.append(np.full(len(xyz), path_idx, dtype=np.int64))
            path_idx += 1

    ang_chunks_xyz, ang_chunks_idx = [], []
    for lvl, paths in angle_paths:
        for xy in paths:
            z = surface.mag_at(xy[:, 1], xy[:, 0])
            # Cell 9: angle contours drop vertices where |zeta| > 6 BEFORE cutout.
            mask = z <= z_limit
            xy_kept = xy[mask]
            z_kept = z[mask]
            if len(xy_kept) < 2:
                continue
            xyz = np.column_stack([xy_kept, z_kept])
            keep = _cutout_kept(xyz)
            xyz = xyz[keep]
            if len(xyz) < 2:
                continue
            ang_chunks_xyz.append(xyz)
            ang_chunks_idx.append(np.full(len(xyz), path_idx, dtype=np.int64))
            path_idx += 1

    # Peak contours: cell 10 — same extractor as magnitude, additionally
    # filtered to |imag| < 5 (focuses detail on the s=1 pole region).
    peak_chunks_xyz, peak_chunks_idx = [], []
    for lvl, paths in peak_paths:
        for xy in paths:
            z = np.full(len(xy), lvl)
            xyz = np.column_stack([xy, z])
            keep = _cutout_kept(xyz)
            xyz = xyz[keep]
            xyz = xyz[(xyz[:, 0] < 5) & (xyz[:, 0] > -5)]
            if len(xyz) < 2:
                continue
            peak_chunks_xyz.append(xyz)
            peak_chunks_idx.append(np.full(len(xyz), path_idx, dtype=np.int64))
            path_idx += 1

    timer.tick("top_xyz boundary curves")
    # Cell 11's top_xyz: sample the actual surface (|zeta|) along the cutout
    # polygon perimeter so the buffer has good coverage AT the polygon edges.
    # This is the piece I'd been missing entirely. Without it the buffer
    # near the cutout boundary doesn't track the real surface height.
    # Find boundary cutoffs the same way the notebook does.
    eps = 0.01
    top_rear_cutoff = -2.0
    while surface.mag_at(top_rear_cutoff, 28.0) > z_limit:
        top_rear_cutoff += eps
    top_rear_cutoff -= eps
    top_front_cutoff = 10.0
    while surface.mag_at(r_min, top_front_cutoff) > z_limit:
        top_front_cutoff -= eps
    top_front_cutoff += eps

    def sample_curve(im_arr, re_arr):
        zs = surface.mag_at(re_arr, im_arr)
        return np.column_stack([im_arr, re_arr, zs])

    top_xyz_chunks, top_idx_chunks = [], []

    def add_curve(im_arr, re_arr):
        nonlocal path_idx
        xyz = sample_curve(im_arr, re_arr)
        if len(xyz) >= 2:
            top_xyz_chunks.append(xyz)
            top_idx_chunks.append(np.full(len(xyz), path_idx, dtype=np.int64))
            path_idx += 1

    # 9 boundary curves from cell 11
    add_curve(np.linspace(0, top_front_cutoff, 100), np.full(100, r_min))
    add_curve(np.linspace(0, -5, 100), np.full(100, -2.0))
    add_curve(np.full(100, -5.0), np.linspace(-2, -0.5, 100))
    add_curve(np.linspace(-5, -14, 100), np.full(100, -0.5))
    add_curve(np.full(100, -14.0), np.linspace(0.5, -0.5, 100))
    add_curve(np.linspace(-14, -28, 100), np.full(100, 0.5))
    add_curve(np.full(100, -28.0), np.linspace(0.5, r_max, 100))
    add_curve(np.linspace(-28, 28, 1000), np.full(1000, r_max))
    add_curve(np.full(100, 28.0), np.linspace(top_rear_cutoff, r_max, 100))

    # Concatenate everything into major_data
    all_chunks_xyz = mag_chunks_xyz + ang_chunks_xyz + peak_chunks_xyz + top_xyz_chunks
    all_chunks_idx = mag_chunks_idx + ang_chunks_idx + peak_chunks_idx + top_idx_chunks
    major_data = np.concatenate(all_chunks_xyz)
    major_indices = np.concatenate(all_chunks_idx)
    print(f"      {len(major_data)} total contour+boundary vertices "
          f"across {path_idx} paths")

    # Stratum-bump indices (cell 11) so paths spanning imag boundaries split.
    BUMP = 1_000_000
    major_indices = major_indices.copy()
    major_indices[major_data[:, 0] < 14] += BUMP
    major_indices[major_data[:, 0] < 5] += BUMP
    major_indices[major_data[:, 0] < 0] += BUMP

    # Projection per cell 11
    isometric_scale_factor = -0.18
    real_scale_factor = 0.75
    x_angle = -79.5  # zeta2 value; cell 11 of zeta.ipynb showed 0 but the saved output uses ~-79.5
    z_angle = -90
    rx = R.from_euler("x", x_angle, degrees=True)
    rz = R.from_euler("z", z_angle, degrees=True)

    def project(xyz):
        out = xyz.copy()
        out[:, 1] *= real_scale_factor
        out[:, 2] = np.minimum(z_limit, out[:, 2])
        out[:, 0] *= -1
        out[:, 1] -= isometric_scale_factor * out[:, 0]
        return rx.apply(rz.apply(out))

    timer.tick("project major_data")
    major_rotated = project(major_data)

    timer.tick("Delaunay")
    # Cell 14 takes Delaunay of the FIRST TWO columns of all_data (i.e., the
    # ORIGINAL imag/real coords) and rasterizes using x_values/y_values from
    # the rotated coords. This is the bizarre buffer construction the notebook
    # uses to get the cutout-shaped buffer naturally.
    tri = Delaunay(major_data[:, :2])
    print(f"      {len(tri.simplices)} simplices")

    x_values = major_rotated[:, 0]
    y_values = major_rotated[:, 1]
    z_values = major_rotated[:, 2]

    zb = ZBuffer(x_values.min(), x_values.max(),
                 y_values.min(), y_values.max(), buffer_shape)

    if args.gpu:
        timer.tick("rasterize (gpu)")
        rasterize_triangles_gpu(zb, tri.simplices, x_values, y_values, z_values)
    else:
        timer.tick("rasterize")
        rasterize_triangles(zb, tri.simplices, x_values, y_values, z_values,
                            progress=progress)

    timer.tick("clip contours")
    segments = clip_hidden_lines(zb, major_rotated, major_indices,
                                 margin=args.clip_margin)

    timer.tick("boundary + shading lines")
    # base_xy_major: cutout polygon walls + vertical corners (cell 11)
    base_xyz_major_3d = [
        np.array([[28., r_min, z_limit], [28., r_min, 0.]]),
        np.array([[28., r_min, 0.], [0., r_min, 0.]]),
        np.array([[28., r_min, z_limit], [28., top_rear_cutoff, z_limit]]),
        np.array([[28., r_min, z_limit], [top_front_cutoff, r_min, z_limit]]),
        np.array([[0., r_min, 0.], [0., -2., 0.]]),
        np.array([[0., -2., 0.], [-5., -2., 0.]]),
        np.array([[-5., -2., 0.], [-5., -0.5, 0.]]),
        np.array([[-5., -0.5, 0.], [-14., -0.5, 0.]]),
        np.array([[-14., -0.5, 0.], [-14., 0.5, 0.]]),
        np.array([[-14., 0.5, 0.], [-28., 0.5, 0.]]),
        np.array([[-28., 0.5, 0.], [-28., r_max, 0.]]),
        np.array([[-5., -2., 0.], [-5., -2., surface.mag_at(-2., -5.)]]),
        np.array([[-5., -0.5, 0.], [-5., -0.5, surface.mag_at(-0.5, -5.)]]),
        np.array([[-14., -0.5, 0.], [-14., -0.5, surface.mag_at(-0.5, -14.)]]),
        np.array([[-14., 0.5, 0.], [-14., 0.5, surface.mag_at(0.5, -14.)]]),
        np.array([[-28., 0.5, 0.], [-28., 0.5, surface.mag_at(0.5, -28.)]]),
        np.array([[-28., r_max, 0.], [-28., r_max, surface.mag_at(r_max, -28.)]]),
    ]
    base_xy_major = [project(seg)[:, :2] for seg in base_xyz_major_3d]

    # base_contours: vertical hatch curtains from ground to surface, sampled
    # along the cutout polygon edges (cell 11). One wall_hatch per edge — the
    # first four run along imag (const real), the last three along real.
    base_density = 6
    s1 = np.linspace(28, 0, int(28 * base_density))[1:-1]
    s2 = np.linspace(0, -5, int(5 * base_density))[1:-1]
    s3 = np.linspace(-5, -14, int(9 * base_density))[1:-1]
    s4 = np.linspace(-14, -28, int(14 * base_density))[1:-1]
    s5 = np.linspace(-2, -0.5, int(2 * 2.5 * base_density))[1:-1]
    s6 = np.linspace(-0.5, 0.5, int(2 * base_density))[1:-1]
    s7 = np.linspace(0.5, r_max, int(2 * (r_max - 0.5) * base_density))[1:-1]
    base_contours_xy = (
        wall_hatch(s1, np.full_like(s1, r_min), surface, project)
        + wall_hatch(s2, np.full_like(s2, -2.0), surface, project)
        + wall_hatch(s3, np.full_like(s3, -0.5), surface, project)
        + wall_hatch(s4, np.full_like(s4, 0.5), surface, project)
        + wall_hatch(np.full_like(s5, -5.0), s5, surface, project)
        + wall_hatch(np.full_like(s6, -14.0), s6, surface, project)
        + wall_hatch(np.full_like(s7, -28.0), s7, surface, project)
    )

    # top_contours: horizontal hatching at z=6 marking the s=1 pole cap (cell 11)
    top_contours_3d = []
    for im in np.linspace(28, top_front_cutoff,
                          max(2, int((28 - top_front_cutoff) * base_density // 2)))[1:-1]:
        cutoff = r_min
        while surface.mag_at(cutoff, im) > z_limit:
            cutoff += eps
        top_contours_3d.append(np.array([[im, r_min, z_limit],
                                         [im, cutoff - eps, z_limit]]))
    top_contours_xy = [project(seg)[:, :2] for seg in top_contours_3d]

    print(f"      {len(base_xy_major)} polygon lines, "
          f"{len(base_contours_xy)} vertical, "
          f"{len(top_contours_xy)} top")

    timer.tick("save hi_res.svg")
    fig, ax = plt.subplots(figsize=(16, 16))
    for xy in segments:
        ax.plot(xy[:, 0], xy[:, 1], lw=0.4, c="k")
    for xy in base_xy_major:
        ax.plot(*xy.T, c="k", lw=0.3)
    for xy in base_contours_xy:
        ax.plot(*xy.T, c="k", lw=0.3)
    for xy in top_contours_xy:
        ax.plot(*xy.T, c="k", lw=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(f"{args.output_prefix}_hi_res.svg")
    plt.close(fig)
    timer.done()
    print(f"Wrote {args.output_prefix}_hi_res.svg.")


if __name__ == "__main__":
    main()
