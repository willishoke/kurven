"""Jacobi elliptic cn analytic landscape.

Direct port of 2401/elliptic.ipynb. The notebook samples cn over one
fundamental tile [-K, 0] × [-K', 0] (m=0.64), then tiles 3×6 with
reflections to fill the visible plate; each tile produces a cone-shaped
spire at its singularity corner.

Uses kurven for rasterize/clip, follows the notebook's pipeline:
contour extraction → tile via reflection → angle_major_zeros along
periodic K_prime/K lines → Delaunay-of-vertices buffer → GPU rasterize
→ clip → boundary/base/top geometry.

Run:
    python examples/elliptic.py --gpu
"""

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R

import contourpy

from kurven.contours import _stitch_chunk_seams
from kurven.outline import clip_hidden_lines
from kurven.zbuffer import (
    ZBuffer,
    rasterize_triangles,
    rasterize_triangles_gpu,
)


_TIMINGS = []
_LAST_TICK = [None]


def _tick(name):
    now = time.perf_counter()
    if _LAST_TICK[0] is not None:
        dt = now - _LAST_TICK[0][1]
        _TIMINGS.append((_LAST_TICK[0][0], dt))
        print(f"  [{_LAST_TICK[0][0]:>26s}] {dt:7.2f}s", flush=True)
    _LAST_TICK[0] = (name, now)


def _tick_done():
    _tick("__end__")
    _TIMINGS.pop()
    total = sum(dt for _, dt in _TIMINGS)
    print("  " + "-" * 40)
    for name, dt in _TIMINGS:
        print(f"  [{name:>26s}] {dt:7.2f}s  {100*dt/total:4.1f}%")
    print(f"  [{'total':>26s}] {total:7.2f}s")


def _cn(z, m):
    """Jacobi cn via scipy.special.ellipj. Accepts complex z."""
    # ellipj returns (sn, cn, dn, ph)
    return sp.ellipj(z.real if np.iscomplexobj(z) else z, m)[1] \
        if not np.iscomplexobj(z) else _ellipj_complex_cn(z, m)


def _ellipj_complex_cn(z, m):
    """cn for complex z using the addition formula. scipy's ellipj only
    handles real arguments; this wraps via the standard identity:
        cn(u + iv) = (cn(u, m) * cn(v, 1-m) - i * sn(u, m) * dn(u, m) * sn(v, 1-m) * dn(v, 1-m))
                     / (1 - dn(u, m)**2 * sn(v, 1-m)**2)
    """
    u, v = z.real, z.imag
    m1 = 1.0 - m
    sn_u, cn_u, dn_u, _ = sp.ellipj(u, m)
    sn_v, cn_v, dn_v, _ = sp.ellipj(v, m1)
    denom = cn_v**2 + m * sn_u**2 * sn_v**2
    num_re = cn_u * cn_v
    num_im = -sn_u * dn_u * sn_v * dn_v
    return (num_re + 1j * num_im) / denom


def _contour_per_level(array, levels, real_bounds, imag_bounds):
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    n_real, n_imag = array.shape
    gen = contourpy.contour_generator(
        z=array, name="threaded",
        line_type=contourpy.LineType.Separate,
        chunk_count=max(1, os.cpu_count() or 1),
    )
    out = []
    for lvl in levels:
        segs = _stitch_chunk_seams(gen.lines(float(lvl)))
        converted = []
        for seg in segs:
            if len(seg) < 2:
                continue
            xy = seg.copy()
            xy[:, 0] *= (i_max - i_min) / n_imag
            xy[:, 0] += i_min
            xy[:, 1] *= (r_max - r_min) / n_real
            xy[:, 1] += r_min
            converted.append(xy)
        out.append((float(lvl), converted))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=2000,
                        help="Sample resolution per tile (notebook used 5000)")
    parser.add_argument("--buffer", type=int, default=6000)
    parser.add_argument("--output-prefix", type=str, default="elliptic")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--clip-margin", type=float, default=0.01)
    args = parser.parse_args()

    if args.backend:
        matplotlib.use(args.backend)
    buffer_shape = (args.buffer, args.buffer)
    progress = not args.no_progress

    # Constants from cell 1
    m = 0.8 ** 2
    K = float(sp.ellipk(m))
    K_prime = float(sp.ellipk(1 - m))
    print(f"m={m}, K={K:.4f}, K'={K_prime:.4f}")
    z_limit = 4.0
    eps = 1e-5
    r_min, r_max = -K + eps, 0 - eps
    i_min, i_max = -K_prime + eps, 0 - eps

    # Effective K, K' after the epsilon shrink (matches notebook cell 1's
    # `K -= 2*epsilon` step so the tile boundaries land on the same coords
    # as the explicit base_xyz_major geometry below).
    K_eff = K - 2 * eps
    K_prime_eff = K_prime - 2 * eps

    _tick("sample tile")
    res = args.res
    real = np.linspace(r_min, r_max, res)
    imag = np.linspace(i_min, i_max, res)
    grid = real[:, None] + 1j * imag
    cn = _ellipj_complex_cn(grid, m)
    mag = np.abs(cn)
    angle = np.angle(cn)
    print(f"      tile: shape={mag.shape}, |cn| ∈ [{mag.min():.3f}, {mag.max():.3f}]")

    # Contour levels per cell 1
    mag_major_levels = np.arange(0.2, 4.1, 0.2)
    mag_minor_levels = np.setdiff1d(np.arange(0, 4 + 1e-6, 0.05), mag_major_levels)
    ang_major_levels = np.linspace(-np.pi / 2, 0, 6)
    ang_minor_levels = np.setdiff1d(np.linspace(-np.pi / 2, 0, 21), ang_major_levels)

    _tick("contour generation")
    mag_major_paths = _contour_per_level(mag, mag_major_levels, (r_min, r_max), (i_min, i_max))
    mag_minor_paths = _contour_per_level(mag, mag_minor_levels, (r_min, r_max), (i_min, i_max))
    ang_major_paths = _contour_per_level(angle, ang_major_levels, (r_min, r_max), (i_min, i_max))
    ang_minor_paths = _contour_per_level(angle, ang_minor_levels, (r_min, r_max), (i_min, i_max))

    _tick("build tile data")
    # For each path, build xyz where z = |cn| at vertex (clipped at 4), and
    # tag with a global index. Matches cell 3's `extractContours`.
    def paths_to_xyz_idx(paths, start_idx):
        chunks_xyz, chunks_idx = [], []
        path_idx = start_idx
        for _, segs in paths:
            for xy in segs:
                if len(xy) < 2:
                    continue
                z = np.minimum(
                    z_limit,
                    np.abs(_ellipj_complex_cn(xy[:, 1] + 1j * xy[:, 0], m)),
                )
                xyz = np.column_stack([xy, z])
                chunks_xyz.append(xyz)
                chunks_idx.append(np.full(len(xy), path_idx, dtype=np.int64))
                path_idx += 1
        if not chunks_xyz:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), path_idx
        return (
            np.concatenate(chunks_xyz),
            np.concatenate(chunks_idx),
            path_idx,
        )

    p = 0
    mag_major_tile, mag_major_idx_tile, p = paths_to_xyz_idx(mag_major_paths, p)
    mag_minor_tile, mag_minor_idx_tile, p = paths_to_xyz_idx(mag_minor_paths, p)
    ang_major_tile, ang_major_idx_tile, p = paths_to_xyz_idx(ang_major_paths, p)
    ang_minor_tile, ang_minor_idx_tile, p = paths_to_xyz_idx(ang_minor_paths, p)

    _tick("tile 3x6 with reflections")
    # Reproduce cell 3's tiling: 3 columns (imag direction) × 6 rows (real
    # direction). Even columns are reflected across the imag axis; even rows
    # are reflected across the real axis.
    def tile_data(tile_xyz, tile_idx):
        max_orig_idx = int(tile_idx.max()) + 1 if len(tile_idx) else 0
        out_xyz = [tile_xyz]
        out_idx = [tile_idx]
        for i in range(3):
            for r in range(6):
                data = tile_xyz.copy()
                if not (i % 2):
                    data[:, 0] *= -1
                    data[:, 0] += i * K_prime_eff
                else:
                    data[:, 0] += 2 * i * K_prime_eff
                if (r % 2):
                    data[:, 1] *= -1
                    data[:, 1] += (r - 1) * K_eff
                else:
                    data[:, 1] += r * K_eff
                out_xyz.append(data)
                k = 5 * i + 1 + r
                out_idx.append(tile_idx + k * max_orig_idx + k)
        return np.vstack(out_xyz), np.hstack(out_idx)

    mag_major_data, mag_major_indices = tile_data(mag_major_tile, mag_major_idx_tile)
    mag_minor_data, mag_minor_indices = tile_data(mag_minor_tile, mag_minor_idx_tile)
    ang_major_data, ang_major_indices = tile_data(ang_major_tile, ang_major_idx_tile)
    ang_minor_data, ang_minor_indices = tile_data(ang_minor_tile, ang_minor_idx_tile)

    # Cell 3: snap magnitude major z to the nearest level
    if len(mag_major_data):
        diff = np.abs(mag_major_data[:, 2:3] - mag_major_levels.reshape(1, -1))
        mag_major_data[:, 2] = mag_major_levels[np.argmin(diff, axis=1)]

    _tick("angle_major_zeros")
    # Cell 3: extra angle major contours along K_prime*i (imag axis) and K*r
    # (real axis) for i, r in their respective ranges. These add the radial
    # phase lines through each spire.
    ang_zeros_chunks_xyz, ang_zeros_chunks_idx = [], []
    next_idx = (ang_major_indices.max() + 1) if len(ang_major_indices) else 0
    for i in range(-1, 4):
        re_arr = np.linspace(-K_eff, (i >= 0) * 5 * K_eff, res)
        im_arr = np.full(res, K_prime_eff * i)
        z = np.minimum(z_limit, np.abs(_ellipj_complex_cn(re_arr + 1j * im_arr, m)))
        xyz = np.column_stack([im_arr, re_arr, z])
        idx_per = np.ceil(re_arr / K_eff).astype(np.int64) + 7 * (i + 1) + next_idx
        ang_zeros_chunks_xyz.append(xyz)
        ang_zeros_chunks_idx.append(idx_per)
    for r in range(-1, 6):
        im_arr = np.linspace(-K_prime_eff * (r <= 0), 3 * K_prime_eff, res)
        re_arr = np.full(res, K_eff * r)
        z = np.minimum(z_limit, np.abs(_ellipj_complex_cn(re_arr + 1j * im_arr, m)))
        xyz = np.column_stack([im_arr, re_arr, z])
        # Notebook uses `5*(i+1)` here with `i` left dangling from the prev
        # loop — for our port we just give every (r, im_index) its own tag.
        idx_per = np.ceil(im_arr / K_prime_eff).astype(np.int64) + 1000 + r * 7
        idx_per += next_idx
        ang_zeros_chunks_xyz.append(xyz)
        ang_zeros_chunks_idx.append(idx_per)
    ang_major_data = np.vstack([ang_major_data] + ang_zeros_chunks_xyz)
    ang_major_indices = np.hstack([ang_major_indices] + ang_zeros_chunks_idx)

    _tick("angle minor truncation")
    # Cell 3's diff-and-even-K filter: keep only angle minor vertices on
    # transitions AND not near (even K, odd K_prime) lattice intersections.
    def trim_angle(data, indices, eps=0.01):
        if len(data) == 0:
            return data, indices
        d1 = np.insert(np.diff(data[:, 2]), 0, 0).astype(bool)
        d2 = np.insert(np.diff(data[:, 2]), -1, 0).astype(bool)
        even_K = np.abs(data[:, 1] / (2 * K_eff) - np.round(data[:, 1] / (2 * K_eff))) < eps
        odd_K_prime = np.abs(
            (data[:, 0] + K_prime_eff) / (2 * K_prime_eff)
            - np.round((data[:, 0] + K_prime_eff) / (2 * K_prime_eff))
        ) < eps
        cond = (d1 | d2) & ~(even_K & odd_K_prime)
        return data[cond], indices[cond]

    ang_minor_data, ang_minor_indices = trim_angle(ang_minor_data, ang_minor_indices)
    ang_major_data, ang_major_indices = trim_angle(ang_major_data, ang_major_indices)

    _tick("project")
    # Cell 4 projection
    isometric_scale_factor = 0.51
    x_angle = -63
    z_angle = -90
    rx = R.from_euler("x", x_angle, degrees=True)
    rz = R.from_euler("z", z_angle, degrees=True)

    def project(xyz):
        out = xyz.copy()
        out[:, 0] *= -1
        out[:, 1] -= isometric_scale_factor * out[:, 0]
        return rx.apply(rz.apply(out))

    major_data = np.vstack((mag_major_data, ang_major_data))
    major_indices = np.hstack((mag_major_indices, ang_major_indices))
    minor_data = np.vstack((mag_minor_data, ang_minor_data))
    minor_indices = np.hstack((mag_minor_indices, ang_minor_indices))
    all_data = np.vstack((major_data, minor_data))

    major_rotated = project(major_data)
    minor_rotated = project(minor_data)
    print(f"      major: {len(major_data)} verts, minor: {len(minor_data)} verts")

    _tick("Delaunay")
    tri = Delaunay(all_data[:, :2])
    print(f"      {len(tri.simplices)} simplices")

    _tick("rasterize buffer")
    # Project all_data and rasterize using projected coordinates.
    all_rotated = np.vstack((major_rotated, minor_rotated))
    ax_val = all_rotated[:, 0]
    ay_val = all_rotated[:, 1]
    az_val = all_rotated[:, 2]
    zb = ZBuffer(ax_val.min(), ax_val.max(),
                 ay_val.min(), ay_val.max(), buffer_shape)
    if args.gpu:
        rasterize_triangles_gpu(zb, tri.simplices, ax_val, ay_val, az_val)
    else:
        rasterize_triangles(zb, tri.simplices, ax_val, ay_val, az_val,
                            progress=progress)

    _tick("clip major")
    major_segs = clip_hidden_lines(zb, major_rotated, major_indices,
                                   margin=args.clip_margin)
    minor_segs = clip_hidden_lines(zb, minor_rotated, minor_indices,
                                   margin=args.clip_margin)

    _tick("boundary geometry")
    # Cell 4: explicit cutout polygon walls + spire-base + spire-top hatching.
    # All values verbatim from cell 4 (with scipy.special.ellipj as the
    # function evaluator).
    def cn_mag(re_v, im_v):
        return float(np.abs(_ellipj_complex_cn(np.array([re_v + 1j * im_v]), m)[0]))

    eps_b = 0.005
    base_xyz_major_3d = [
        np.array([[-K_prime, -K, 0.0], [-K_prime, 0.0, 0.0]]),
        np.array([[-K_prime, -K - eps_b, 0.0],
                  [-K_prime, -K - eps_b, cn_mag(-K, -K_prime)]]),
        np.array([[-K_prime, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 5 * K, 0.0]]),
        np.array([[0.0, 5 * K, 0.0], [3 * K_prime, 5 * K, 0.0]]),
        np.array([[-K_prime, -K, 0.0],
                  [-K_prime, -K, cn_mag(-K, -K_prime)]]),
        np.array([[-K_prime, 0.0, 0.0], [-K_prime, 0.0, z_limit]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, cn_mag(0.0, 0.0)]]),
        np.array([[3 * K_prime, 5 * K, 0.0],
                  [3 * K_prime, 5 * K, cn_mag(5 * K, 3 * K_prime)]]),
    ]

    # Find real_radius and imag_radius (the spire cap radii)
    eps_r = 0.05
    real_radius = 0.0
    while cn_mag(real_radius, -K_prime) > z_limit:
        real_radius += eps_r
    real_radius -= eps_r
    base_xyz_major_3d.append(np.array([[-K_prime, 0.0, z_limit],
                                       [-K_prime, -real_radius, z_limit]]))

    imag_radius = 0.0
    while cn_mag(0.0, -(K_prime + imag_radius)) > z_limit:
        imag_radius += eps_r
    imag_radius -= eps_r
    base_xyz_major_3d.append(np.array([[-K_prime, 0.0, z_limit],
                                       [-K_prime + imag_radius, 0.0, z_limit]]))

    base_xyz_major_3d.append(np.array([[3 * K_prime, -real_radius, z_limit],
                                       [3 * K_prime, real_radius, z_limit]]))
    base_xyz_major_3d.append(np.array([[3 * K_prime, 2 * K - real_radius, z_limit],
                                       [3 * K_prime, 2 * K + real_radius, z_limit]]))
    base_xyz_major_3d.append(np.array([[3 * K_prime, 4 * K - real_radius, z_limit],
                                       [3 * K_prime, 4 * K + real_radius, z_limit]]))

    base_xy_major = [project(seg)[:, :2] for seg in base_xyz_major_3d]

    # base_contours: vertical hatching from ground to surface along the
    # cutout perimeter (per cell 4)
    base_contour_density = 40
    base_contours_3d = []

    for i in np.linspace(-K, 0, base_contour_density)[1:-1]:
        h = -3 * eps_b + min(z_limit, cn_mag(i, -K_prime))
        base_contours_3d.append(np.array([[-K_prime, i, eps_b],
                                          [-K_prime, i, h]]))
    for i in np.linspace(-K_prime, 0, base_contour_density)[1:-1]:
        h = -eps_b + min(z_limit, cn_mag(0.0, i))
        base_contours_3d.append(np.array([[i, 0.0, eps_b],
                                          [i, 0.0, h]]))
    for i in np.linspace(0, 5 * K, 5 * base_contour_density)[1:-1]:
        h = -eps_b + min(z_limit, cn_mag(i, 0.0))
        base_contours_3d.append(np.array([[0.0, i, eps_b],
                                          [0.0, i, h]]))
    for i in np.linspace(0, 3 * K_prime, 3 * base_contour_density)[1:-1]:
        h = -eps_b + min(z_limit, cn_mag(5 * K, i))
        base_contours_3d.append(np.array([[i, 5 * K, eps_b],
                                          [i, 5 * K, h]]))

    base_contours_xy = [project(seg)[:, :2] for seg in base_contours_3d]

    # top_contours: horizontal hatching at z=4 marking each spire's cap
    top_contour_density = 16
    top_contours_3d = []
    eps_t = 0.001
    for c in range(top_contour_density // 2):
        i = (2 * c / top_contour_density) * imag_radius - K_prime
        r = 0.0
        while cn_mag(r, -i) > z_limit:
            r -= eps_t
        r += eps_t
        top_contours_3d.append(np.array([[i, eps_t, z_limit], [i, r, z_limit]]))
        for j in range(3):
            top_contours_3d.append(np.array([[i + 2 * K_prime, j * 2 * K - r, z_limit],
                                             [i + 2 * K_prime, j * 2 * K + r, z_limit]]))
            top_contours_3d.append(np.array([[-i, j * 2 * K - r, z_limit],
                                             [-i, j * 2 * K + r, z_limit]]))
            top_contours_3d.append(np.array([[-i + 2 * K_prime, j * 2 * K - r, z_limit],
                                             [-i + 2 * K_prime, j * 2 * K + r, z_limit]]))
    top_contours_xy = [project(seg)[:, :2] for seg in top_contours_3d]

    print(f"      {len(base_xy_major)} polygon, "
          f"{len(base_contours_xy)} vertical, "
          f"{len(top_contours_xy)} top")

    _tick("save hi_res.svg")
    fig, ax = plt.subplots(figsize=(16, 12))
    for xy in major_segs:
        ax.plot(xy[:, 0], xy[:, 1], lw=0.3, c="k")
    for xy in minor_segs:
        ax.plot(xy[:, 0], xy[:, 1], lw=0.1, c="k")
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
    _tick_done()
    print(f"Wrote {args.output_prefix}_hi_res.svg.")


if __name__ == "__main__":
    main()
