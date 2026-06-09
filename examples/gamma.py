"""1:1 reproduction of the original Jahnke-Emde gamma plate.

Mirrors the notebook cell-by-cell, but using the `kurven` modules for the
generic stages (contour extraction, projection, rasterization, outline
extraction, hidden-line clipping). The threshold/truncation and pole-cutoff
logic — cell 8 of the original — stays here because it's specific to the
gamma function's pole structure.

Run at full resolution:
    python examples/gamma.py

Smoke-test at lower resolution:
    python examples/gamma.py --res 800 --buffer 1600
"""

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


_TIMINGS = []
_LAST_TICK = [None]


def _tick(name):
    """Close the previous timing interval, start a new one. Call between phases."""
    now = time.perf_counter()
    if _LAST_TICK[0] is not None:
        dt = now - _LAST_TICK[0][1]
        _TIMINGS.append((_LAST_TICK[0][0], dt))
        print(f"  [{_LAST_TICK[0][0]:>26s}] {dt:7.2f}s", flush=True)
    _LAST_TICK[0] = (name, now)


def _tick_done():
    _tick("__end__")
    _TIMINGS.pop()  # drop the dummy
    total = sum(dt for _, dt in _TIMINGS)
    print("  " + "-" * 40)
    for name, dt in _TIMINGS:
        print(f"  [{name:>26s}] {dt:7.2f}s  {100*dt/total:4.1f}%")
    print(f"  [{'total':>26s}] {total:7.2f}s")

from kurven.contours import (
    contour_adaptive,
    decimate_outside_critical_zone,
    extract_contours,
    group_by_index,
    mirror_x,
)
from kurven.outline import clip_hidden_lines, extract_outline
from kurven.sampling import gradient_zones, sample_adaptive
from kurven.zbuffer import (
    ZBuffer,
    rasterize_triangles,
    rasterize_triangles_gpu,
    surface_grid_mesh,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=10000,
                        help="Effective fine resolution (matches original uniform 10000²)")
    parser.add_argument("--buffer", type=int, default=20000)
    parser.add_argument("--output-prefix", type=str, default="gamma")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--backend", type=str, default=None,
                        help="Matplotlib backend (e.g. 'Agg' for headless)")
    parser.add_argument("--adaptive", dest="adaptive", action="store_true", default=True,
                        help="Use multi-resolution sampling (default)")
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                        help="Use single-pass uniform sampling at --res")
    parser.add_argument("--coarse-res", type=int, default=2000,
                        help="Resolution outside high-gradient zones (adaptive only)")
    parser.add_argument("--probe-res", type=int, default=400,
                        help="Resolution for gradient-zone discovery")
    parser.add_argument("--zone-pad", type=float, default=0.05,
                        help="Fractional outward padding on each gradient-discovered fine zone. Larger = seam falls in calmer territory.")
    parser.add_argument("--surface-res", type=int, default=1000,
                        help="Surface mesh density for depth buffer")
    parser.add_argument("--clip-margin", type=float, default=0.01,
                        help="Z-margin for hidden-line clipping (tiny eps with correct mesh)")
    parser.add_argument("--stitch-tolerance", type=float, default=None,
                        help="Weld same-level coarse/fine contour paths whose endpoints meet within this distance near a zone boundary. Default: ~1 coarse cell.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use moderngl-backed GPU rasterizer (much faster). Requires `pip install kurven[gpu]`.")
    args = parser.parse_args()

    if args.backend:
        matplotlib.use(args.backend)

    res = args.res
    buffer_shape = (args.buffer, args.buffer)
    progress = not args.no_progress

    r_min, r_max = -4.5, 4.5
    i_min, i_max = 0.0001, 2.5
    real_bounds = (r_min, r_max)
    imag_bounds = (i_min, i_max)

    real = np.linspace(r_min, r_max, res)
    imag = np.linspace(i_min, i_max, res)

    min_log_value = -4
    max_log_value = 1
    angle_major_division = 12
    magnitude_major_division = 48
    minor_division = 10

    angle_major_interval = np.linspace(-np.pi, np.pi, 1 + angle_major_division)
    angle_minor_interval = np.setdiff1d(
        np.linspace(-np.pi, np.pi, 1 + angle_major_division * minor_division),
        angle_major_interval,
    )
    magnitude_major_interval = np.array([
        .0003, 0.001, .003, 0.01, .03, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4,
        1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
    ])
    magnitude_minor_interval = np.logspace(
        min_log_value, max_log_value,
        1 + magnitude_major_division * minor_division, base=10,
    )
    magnitude_minor_interval = np.array([
        x for x in magnitude_minor_interval
        if np.all(np.abs(magnitude_major_interval - x) > 1e-6)
    ])

    def height(z):
        return np.abs(scipy.special.gamma(z))

    if args.adaptive:
        _tick("gradient zones")
        fine_zones = gradient_zones(
            scipy.special.gamma, real_bounds, imag_bounds,
            probe_res=args.probe_res, pad=args.zone_pad,
        )
        print(f"      {len(fine_zones)} fine zone(s):")
        for zr_lo, zr_hi, zi_lo, zi_hi in fine_zones:
            print(f"        real=[{zr_lo:+.2f},{zr_hi:+.2f}] imag=[{zi_lo:.3f},{zi_hi:.3f}]")

        _tick("sample adaptive")
        samples = sample_adaptive(
            scipy.special.gamma, real_bounds, imag_bounds,
            coarse_res=args.coarse_res, fine_res=res, fine_zones=fine_zones,
        )

        if args.stitch_tolerance is None:
            coarse_cell = max((r_max - r_min) / args.coarse_res, (i_max - i_min) / args.coarse_res)
            stitch_tol = 1.5 * coarse_cell
        else:
            stitch_tol = args.stitch_tolerance

        _tick("contour_adaptive x4")
        ang_major_xyz, ang_major_idx = contour_adaptive(
            samples, angle_major_interval, height,
            contour_op=np.angle, stitch_tolerance=stitch_tol,
        )
        ang_minor_xyz, ang_minor_idx = contour_adaptive(
            samples, angle_minor_interval, height,
            contour_op=np.angle, stitch_tolerance=stitch_tol,
        )
        mag_major_xyz, mag_major_idx = contour_adaptive(
            samples, magnitude_major_interval, height,
            contour_op=np.abs, stitch_tolerance=stitch_tol,
        )
        mag_minor_xyz, mag_minor_idx = contour_adaptive(
            samples, magnitude_minor_interval, height,
            contour_op=np.abs, stitch_tolerance=stitch_tol,
        )
    else:
        _tick("eval grid")
        grid = real[:, None] + 1j * imag
        gamma = scipy.special.gamma(grid)
        magnitude = np.abs(gamma)
        angle = np.angle(gamma)

        _tick("mpl.contour x4")
        fig_cs, ax_cs = plt.subplots()
        try:
            mag_major_cs = ax_cs.contour(magnitude, levels=magnitude_major_interval)
            mag_minor_cs = ax_cs.contour(magnitude, levels=magnitude_minor_interval)
            ang_major_cs = ax_cs.contour(angle, levels=angle_major_interval)
            ang_minor_cs = ax_cs.contour(angle, levels=angle_minor_interval)
        finally:
            plt.close(fig_cs)

        _tick("extract_contours x4")
        ang_major_xyz, ang_major_idx = extract_contours(
            ang_major_cs, height, real_bounds, imag_bounds, res
        )
        ang_minor_xyz, ang_minor_idx = extract_contours(
            ang_minor_cs, height, real_bounds, imag_bounds, res
        )
        mag_major_xyz, mag_major_idx = extract_contours(
            mag_major_cs, height, real_bounds, imag_bounds, res
        )
        mag_minor_xyz, mag_minor_idx = extract_contours(
            mag_minor_cs, height, real_bounds, imag_bounds, res
        )

    _tick("decimate + mirror")
    centerline_xy = np.vstack((np.zeros((1, res)), np.linspace(r_min, r_max, res)))
    centerline_z = np.abs(scipy.special.gamma(centerline_xy[1] + centerline_xy[0] * 1j))
    centerline_xyz = np.vstack((centerline_xy, centerline_z.reshape(-1))).T
    centerline_idx = np.ceil(centerline_xyz[:, 1]).astype(np.int64)

    fine_threshold = 1.3
    coarse_interval = 20
    z_range = magnitude_major_interval[[-1, -4, -6, -7]][::-1]

    ang_major_xyz, ang_major_idx = decimate_outside_critical_zone(
        ang_major_xyz, ang_major_idx,
        fine_threshold=fine_threshold, coarse_interval=coarse_interval,
    )
    ang_major_xyz, ang_major_idx = mirror_x(
        ang_major_xyz, ang_major_idx, (centerline_xyz, centerline_idx)
    )

    ang_minor_xyz, ang_minor_idx = decimate_outside_critical_zone(
        ang_minor_xyz, ang_minor_idx,
        fine_threshold=fine_threshold, coarse_interval=coarse_interval,
    )
    ang_minor_xyz, ang_minor_idx = mirror_x(ang_minor_xyz, ang_minor_idx)

    mag_major_xyz, mag_major_idx = mag_major_xyz[::20], mag_major_idx[::20]
    mag_major_xyz, mag_major_idx = mirror_x(mag_major_xyz, mag_major_idx)

    mag_minor_xyz, mag_minor_idx = mag_minor_xyz[::20], mag_minor_idx[::20]
    mag_minor_xyz, mag_minor_idx = mirror_x(mag_minor_xyz, mag_minor_idx)

    _tick("top-of-pole contours")
    top_contours = _build_top_pole_contours(
        magnitude_major_interval, real_bounds, imag_bounds, res,
        z_range=z_range,
    )

    _tick("pole-cutoff truncation")
    epsilon = 0.1
    for y, z in zip([-3.5, -2.5, -1.5, .5], z_range):
        cond = ~((mag_major_xyz[:, 1] < y) & (mag_major_xyz[:, 2] > z + epsilon))
        mag_major_xyz = mag_major_xyz[cond]
        mag_major_idx = mag_major_idx[cond]
    for y, z in zip([-3.5, -2.5, -1.5, .5], z_range):
        cond = ~((mag_minor_xyz[:, 1] < y) & (mag_minor_xyz[:, 2] > z + epsilon / 10))
        mag_minor_xyz = mag_minor_xyz[cond]
        mag_minor_idx = mag_minor_idx[cond]

    nan_cond = ~np.isnan(ang_major_xyz[:, 2])
    ang_major_xyz = ang_major_xyz[nan_cond]
    ang_major_idx = ang_major_idx[nan_cond]

    epsilon_values = [.001, .01, .005, .01]
    for y, z, eps in zip([-3.5, -2.5, -1.5, .5], z_range, epsilon_values):
        cond = ((ang_major_xyz[:, 1] < y) & (ang_major_xyz[:, 2] > z))
        ang_major_xyz[cond, 2] = z
        diff1 = np.insert(np.diff(ang_major_xyz[:, 2]), 0, 0).astype(bool)
        diff2 = np.insert(np.diff(ang_major_xyz[:, 2]), -1, 0).astype(bool)
        cond2 = (ang_major_xyz[:, 1] > y) | (
            (diff1 | diff2)
            & (np.linalg.norm(ang_major_xyz[:, :2] - np.round(ang_major_xyz[:, :2]), axis=1) > eps)
        )
        ang_major_xyz = ang_major_xyz[cond2]
        ang_major_idx = ang_major_idx[cond2]

    for y, z, eps in zip([-3.5, -2.5, -1.5, .5], z_range, epsilon_values):
        cond = ((ang_minor_xyz[:, 1] < y) & (ang_minor_xyz[:, 2] > z))
        ang_minor_xyz[cond, 2] = z
        diff1 = np.insert(np.diff(ang_minor_xyz[:, 2]), 0, 0).astype(bool)
        diff2 = np.insert(np.diff(ang_minor_xyz[:, 2]), -1, 0).astype(bool)
        cond2 = (ang_minor_xyz[:, 1] > y) | (
            (diff1 | diff2)
            & (np.abs(ang_minor_xyz[:, 1] - np.round(ang_minor_xyz[:, 1])) > eps)
        )
        ang_minor_xyz = ang_minor_xyz[cond2]
        ang_minor_idx = ang_minor_idx[cond2]

    z_limit = magnitude_major_interval[-4]
    cond = (mag_major_xyz[:, 1] < 0.5) | (mag_major_xyz[:, 2] < z_limit + epsilon)
    mag_major_xyz = mag_major_xyz[cond]
    mag_major_idx = mag_major_idx[cond]
    correction = np.array([
        np.argmin(np.abs(magnitude_major_interval - z)) for z in mag_major_xyz[:, 2]
    ])
    mag_major_xyz[:, 2] = magnitude_major_interval[correction]

    cond = (mag_minor_xyz[:, 1] < 0.5) | (mag_minor_xyz[:, 2] < z_limit + epsilon / 10)
    mag_minor_xyz = mag_minor_xyz[cond]
    mag_minor_idx = mag_minor_idx[cond]

    cond = (ang_major_xyz[:, 1] < 0.5) | (ang_major_xyz[:, 2] < z_limit)
    ang_major_xyz = ang_major_xyz[cond]
    ang_major_idx = ang_major_idx[cond]

    cond = (ang_minor_xyz[:, 1] < 0.5) | (ang_minor_xyz[:, 2] < z_limit)
    ang_minor_xyz = ang_minor_xyz[cond]
    ang_minor_idx = ang_minor_idx[cond]

    _tick("base/back boundary")
    isometric_scale_factor = 0.5
    x_angle = -55
    z_angle = -90
    rx = R.from_euler("x", x_angle, degrees=True)
    rz = R.from_euler("z", z_angle, degrees=True)

    base_xy_minor, base_xy_major, cutoff_point = _build_base_geometry(
        real_bounds, imag_bounds, imag, res, z_limit,
        rx, rz, isometric_scale_factor,
    )

    # Bounds intervals (the four edges of the data box, lifted to z=|gamma|)
    i_interval = 2 * (imag - np.mean(imag))
    r_interval = np.linspace(r_min, cutoff_point, res)
    bounds_intervals = [
        r_min + 1j * i_interval,
        r_min + 1j * i_interval,
        cutoff_point + 1j * i_interval,
        r_interval + 1j * (-i_max),
        r_interval + 1j * i_max,
    ]
    bounds_idx = np.array([
        np.full(x.shape[0], i)
        for x, i in zip(bounds_intervals, range(len(bounds_intervals)))
    ]).flatten()
    bounds = np.array([
        np.clip(np.abs(scipy.special.gamma(x)), 0, z_limit) for x in bounds_intervals
    ])
    bounds[0] = 0
    bounds_xyz_list = [
        np.vstack((np.imag(c), np.real(c), z)) for c, z in zip(bounds_intervals, bounds)
    ]
    bounds_xyz = np.hstack(bounds_xyz_list).T

    major_xyz = np.vstack((mag_major_xyz, ang_major_xyz, bounds_xyz))
    major_idx = np.hstack((mag_major_idx, ang_major_idx, bounds_idx))
    minor_xyz = np.vstack((mag_minor_xyz, ang_minor_xyz))
    minor_idx = np.hstack((mag_minor_idx, ang_minor_idx))
    all_xyz = np.vstack((major_xyz, minor_xyz))

    major_xyz_shear = major_xyz.copy()
    major_xyz_shear[:, 1] += isometric_scale_factor * major_xyz_shear[:, 0]
    major_rotated = rx.apply(rz.apply(major_xyz_shear))

    minor_xyz_shear = minor_xyz.copy()
    minor_xyz_shear[:, 1] += isometric_scale_factor * minor_xyz_shear[:, 0]
    minor_rotated = rx.apply(rz.apply(minor_xyz_shear))

    all_rotated = np.vstack((major_rotated, minor_rotated))
    print(f"      projected shape: {all_rotated.shape}")

    _tick("save raw.svg")
    fig_raw, ax_raw = plt.subplots(figsize=(16, 16))
    for _, xy in group_by_index(major_rotated, major_idx):
        ax_raw.plot(xy[:, 0], xy[:, 1], lw=0.3, c="k")
    ax_raw.plot(*base_xy_major.T, lw=0.3, c="k")
    for c in base_xy_minor:
        ax_raw.plot(*c.T, lw=0.1, c="k")
    for c in top_contours:
        ax_raw.plot(*c.T, lw=0.1, c="k")
    ax_raw.set_aspect("equal")
    ax_raw.axis("off")
    fig_raw.savefig(f"{args.output_prefix}_raw.svg")
    plt.close(fig_raw)

    _tick("surface mesh")
    surf_res = args.surface_res
    surf_real = np.linspace(r_min, r_max, surf_res)
    surf_imag = np.linspace(-i_max, i_max, surf_res)
    surf_grid = surf_real[:, None] + 1j * surf_imag

    # Per-region z caps — must match the contour-truncation caps, otherwise the
    # surface silhouette protrudes above the truncated contours on shorter
    # spires (and the contours protrude above the surface on taller ones).
    # Pole-region bands paired with z_range = [2.5, 3, 4, 5.5] for real bands
    # (<-3.5, <-2.5, <-1.5, <0.5); calm region (real >= 0.5) caps at z_limit.
    surf_caps_1d = np.full_like(surf_real, z_limit)
    surf_caps_1d[surf_real < 0.5] = z_range[3]
    surf_caps_1d[surf_real < -1.5] = z_range[2]
    surf_caps_1d[surf_real < -2.5] = z_range[1]
    surf_caps_1d[surf_real < -3.5] = z_range[0]
    surf_z = np.minimum(np.abs(scipy.special.gamma(surf_grid)), surf_caps_1d[:, None])
    # Vertex (i, j) at (imag[j], real[i], z[i, j]) — matches the (imag, real, z) convention
    surf_vertices, surf_simplices = surface_grid_mesh(
        coords_x=surf_imag, coords_y=surf_real, z_values=surf_z,
    )
    surf_vertices[:, 1] += isometric_scale_factor * surf_vertices[:, 0]
    surf_rotated = rx.apply(rz.apply(surf_vertices))
    print(f"      {len(surf_simplices)} surface triangles")

    sx, sy, sz = surf_rotated.T
    # axis 0 of buffer = x (matches rasterizer's axis0_values=x); axis 1 = y.
    zb = ZBuffer(sx.min(), sx.max(), sy.min(), sy.max(), buffer_shape)

    if args.gpu:
        _tick("rasterize surface (gpu)")
        rasterize_triangles_gpu(zb, surf_simplices, sx, sy, sz)
    else:
        _tick("rasterize surface")
        rasterize_triangles(zb, surf_simplices, sx, sy, sz, progress=progress)

    _tick("outline + clip")
    outline = extract_outline(
        zb,
        cutoff_min=np.array([-4, 0]),
        cutoff_max=np.array([0.8, 5]),
    )

    segments = clip_hidden_lines(zb, major_rotated, major_idx, margin=args.clip_margin)

    _tick("save hi_res.svg")
    fig_final, ax_final = plt.subplots(figsize=(16, 16))
    for xy in segments:
        ax_final.plot(xy[:, 0], xy[:, 1], lw=0.4, c="k")
    ax_final.plot(*base_xy_major.T, lw=0.4, c="k")
    ax_final.plot(*outline.T, lw=0.3, c="k")
    for c in base_xy_minor:
        ax_final.plot(*c.T, lw=0.1, c="k")
    for c in top_contours:
        ax_final.plot(*c.T, lw=0.1, c="k")
    ax_final.set_aspect("equal")
    ax_final.axis("off")
    fig_final.savefig(f"{args.output_prefix}_hi_res.svg")
    plt.close(fig_final)
    _tick_done()
    print(f"Wrote {args.output_prefix}_raw.svg and {args.output_prefix}_hi_res.svg.")


def _build_top_pole_contours(
    magnitude_major_interval, real_bounds, imag_bounds, res, *, z_range,
):
    """Cell 6: shading lines on top of the four pole spires."""
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    isometric_scale_factor = 0.5
    rx = R.from_euler("x", -55, degrees=True)
    rz = R.from_euler("z", -90, degrees=True)
    top_contour_real_scale = 6
    top_contour_imag_scale = 12

    real2 = np.linspace(-r_max, r_max, res)
    imag2 = np.linspace(
        -i_max / top_contour_imag_scale,
        i_max / top_contour_imag_scale,
        res // top_contour_real_scale,
    )
    grid2 = real2[:, None] + 1j * imag2
    magnitude2 = np.abs(scipy.special.gamma(grid2))

    # Compute only the 4 spire-cap levels we use (the original code contoured
    # all 23 and threw away 19). Stay on mpl.contour for this one call because
    # the downstream `path.contains_point` test relies on mpl's grid-edge
    # closure of open contours; contourpy direct leaves them open and the
    # polygon test then returns garbage.
    cutoff_indices = list(reversed([-1, -4, -6, -7]))
    target_levels = [float(magnitude_major_interval[i]) for i in cutoff_indices]
    fig, ax = plt.subplots()
    try:
        mag_cs2 = ax.contour(magnitude2, levels=target_levels)
    finally:
        plt.close(fig)
    if hasattr(mag_cs2, "collections") and mag_cs2.collections:
        cutoff_paths = [list(c.get_paths()) for c in mag_cs2.collections]
    else:
        cutoff_paths = [
            [_pathify(seg) for seg in level_segs]
            for level_segs in mag_cs2.allsegs
        ]

    contours_of_interest = []

    for paths, limit in zip(cutoff_paths, [-3.5, -2.5, -1.5, .5]):
        for path in paths:
            _, y = path.vertices.copy().T
            y *= (r_max - r_min) / res
            y -= (r_max - r_min) / 2
            if (not np.any(y > limit)) and np.any((y < limit) & (y > (limit - (1 + (limit == .5))))):
                contours_of_interest.append(path)

    cs = []
    for p in contours_of_interest:
        l, b = np.min(p.vertices, axis=0)
        r, t = np.max(p.vertices, axis=0)
        for x in np.linspace(l, r, 6 + int((r - l) / 160)):
            run = []
            for y in np.linspace(b, t, 400):
                xy = np.array([x, y])
                if p.contains_point(xy):
                    run.append(xy)
            if len(run) >= 2:
                cs.append(np.array(run))

    top_contours = []
    for c in cs:
        points = c[[0, -1]]
        points[:, 0] -= res / 12
        points[:, 1] -= res / 2
        points[:, 0] *= 2 * (i_max / 12 - i_min / 12) / (res / 6)
        points[:, 1] *= (r_max - r_min) / res
        z = np.abs(scipy.special.gamma(points[0, 1] + 1j * points[0, 0]))
        z = np.full(2, z_range[np.argmin(np.abs(z_range - z))])
        xyz = np.vstack((points.T, z)).T
        xyz[:, 1] += isometric_scale_factor * points[:, 0]
        top_contours.append(rx.apply(rz.apply(xyz))[:, :2])
    return top_contours


def _pathify(seg):
    from matplotlib.path import Path
    return Path(seg)


def _build_base_geometry(
    real_bounds, imag_bounds, imag, res, z_limit,
    rx, rz, isometric_scale_factor,
):
    """Cell 8: base (top edge) and back (left edge) boundary lines."""
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    real_sample_width = 300
    imaginary_sample_width = int(i_max * real_sample_width / (r_max - r_min))
    top_sample_interval = np.linspace(-i_max, i_max, imaginary_sample_width)
    back_sample_interval = np.linspace(-i_max, i_max, 2 * imaginary_sample_width)
    base_xy_minor = []
    cutoff_point = r_max

    for i in top_sample_interval[:-1]:
        r = cutoff_point
        z = np.abs(scipy.special.gamma(r + 1j * i))
        while z > z_limit:
            r -= 0.005
            z = np.abs(scipy.special.gamma(r + 1j * i))
        if cutoff_point == r_max:
            cutoff_point = r + 0.005
        else:
            points = np.array([[i, cutoff_point, z_limit],
                               [i, r + 0.005, z_limit]])
            points[:, 1] += isometric_scale_factor * points[:, 0]
            base_xy_minor.append(rx.apply(rz.apply(points))[:, :2])

    back_threshold = 0.005
    for i in back_sample_interval[:-1]:
        z = np.abs(scipy.special.gamma(r_min + 1j * i))
        if z >= back_threshold:
            points = np.array([[i, r_min, 0],
                               [i, r_min, z]])
            points[:, 1] += isometric_scale_factor * points[:, 0]
            base_xy_minor.append(rx.apply(rz.apply(points))[:, :2])

    base_xyz_major = np.array([
        [i_max, r_min, np.abs(scipy.special.gamma(r_min + 1j * i_max))],
        [i_max, cutoff_point, np.abs(scipy.special.gamma(r_min + 1j * i_max))],
        [i_max, cutoff_point, z_limit],
    ])
    base_xyz_major[:, 1] += isometric_scale_factor * base_xyz_major[:, 0]
    base_xy_major = rx.apply(rz.apply(base_xyz_major))[:, :2]

    base_min_distance = 0.01
    base_sample_interval = np.linspace(r_min, cutoff_point, real_sample_width)
    for r in base_sample_interval:
        points = np.array([
            [i_max, r, np.abs(scipy.special.gamma(r_min + 1j * i_max))],
            [i_max, r, np.minimum(z_limit, np.abs(scipy.special.gamma(r + 1j * i_max)))],
        ])
        if points[1, 2] - points[0, 2] > base_min_distance:
            points[:, 1] += isometric_scale_factor * points[:, 0]
            base_xy_minor.append(rx.apply(rz.apply(points))[:, :2])

    return base_xy_minor, base_xy_major, cutoff_point


if __name__ == "__main__":
    main()
