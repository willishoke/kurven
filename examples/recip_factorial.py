"""Reciprocal gamma analytic landscape — 1/Γ(z).

A fresh example built on the kurven `Surface` noun + the grid-mesh occluder,
not ported from a notebook. Reproduces the classic Jahnke-Emde plate "Relief
der reziproken Fakultät" (Fig. 6): 1/Γ is entire with zeros at z = 0, -1, -2,
...; |1/Γ| ripples along the negative real axis (the zeros), grows without
bound toward the back-left (clamped to a flat cap), and decays toward +Re.

Run:
    python examples/recip_factorial.py --gpu
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
from scipy.spatial.transform import Rotation as R

from kurven.contours import contour_levels
from kurven.occluder import build_occluder, wall_curtain
from kurven.outline import clip_hidden_lines
from kurven.surface import Surface
from kurven.zbuffer import ZBuffer, rasterize_triangles, rasterize_triangles_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1600)
    ap.add_argument("--buffer", type=int, default=6000)
    ap.add_argument("--occluder-res", type=int, default=800)
    ap.add_argument("--output-prefix", default="recip_factorial")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--clip-margin", type=float, default=0.02)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--x-angle", type=float, default=-55.0)
    ap.add_argument("--z-angle", type=float, default=-90.0)
    ap.add_argument("--r-min", type=float, default=-5.5)
    ap.add_argument("--r-max", type=float, default=4.0)
    ap.add_argument("--i-max", type=float, default=2.5)
    ap.add_argument("--z-limit", type=float, default=5.0)
    a = ap.parse_args()
    if not a.gpu:
        matplotlib.use("Agg")

    r_min, r_max = a.r_min, a.r_max
    i_min, i_max = 0.0, a.i_max
    z_limit = a.z_limit
    res = a.res
    real = np.linspace(r_min, r_max, res)
    imag = np.linspace(i_min, i_max, res)
    rb, ib = (r_min, r_max), (i_min, i_max)

    surface = Surface.from_function(ss.rgamma, real, imag, z_limit=z_limit)
    mag, angle = surface.mag, surface.angle
    print(f"|1/Γ| ∈ [{mag.min():.3f}, {mag.max():.1f}], clamp {z_limit}")

    mag_major = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    mag_minor = np.setdiff1d(np.round(np.arange(0.1, z_limit, 0.1), 2), mag_major)
    ang_major = np.linspace(-np.pi, np.pi, 5)                    # every 90 deg
    ang_minor = np.setdiff1d(np.linspace(-np.pi, np.pi, 13), ang_major)  # 30 deg

    def to_xyz(paths, start):
        xs, idx, pi = [], [], start
        for _, segs in paths:
            for xy in segs:
                if len(xy) < 2:
                    continue
                z = surface.height_at(xy[:, 1], xy[:, 0])   # height at (re, im)
                xs.append(np.column_stack([xy, z]))
                idx.append(np.full(len(xy), pi, dtype=np.int64))
                pi += 1
        if not xs:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), pi
        return np.concatenate(xs), np.concatenate(idx), pi

    p = 0
    Mm, Mmi, p = to_xyz(contour_levels(mag, mag_major, rb, ib), p)
    mm, mmi, p = to_xyz(contour_levels(mag, mag_minor, rb, ib), p)
    Am, Ami, p = to_xyz(contour_levels(angle, ang_major, rb, ib), p)
    am, ami, p = to_xyz(contour_levels(angle, ang_minor, rb, ib), p)
    major_data = np.vstack([Mm, Am]); major_idx = np.hstack([Mmi, Ami])
    minor_data = np.vstack([mm, am]); minor_idx = np.hstack([mmi, ami])

    rx = R.from_euler("x", a.x_angle, degrees=True)
    rz = R.from_euler("z", a.z_angle, degrees=True)

    def project(xyz):
        out = xyz.copy()
        out[:, 0] *= -1
        out[:, 1] -= a.scale * out[:, 0]
        return rx.apply(rz.apply(out))

    major_rot = project(major_data)
    minor_rot = project(minor_data)

    # Occluder: clamped heightfield grid + cut-face wall curtains on all four
    # domain edges. No cutout/mask — the displayed region is the full rectangle;
    # the back-left cap falls out of the clamp.
    occ_step = max(1, res // a.occluder_res)
    NW = 600
    edges = [
        (np.full(NW, i_min), np.linspace(r_min, r_max, NW)),   # front (real axis)
        (np.full(NW, i_max), np.linspace(r_min, r_max, NW)),   # back
        (np.linspace(i_min, i_max, NW), np.full(NW, r_min)),   # left
        (np.linspace(i_min, i_max, NW), np.full(NW, r_max)),   # right
    ]
    walls = [wall_curtain(im_a, re_a, surface) for im_a, re_a in edges]
    occ_verts, occ_tris = build_occluder(surface, occ_step, walls=walls)
    print(f"occluder: {len(occ_verts)} verts, {len(occ_tris)} tris")

    occ_rot = project(occ_verts)
    ox, oy, oz = occ_rot[:, 0], occ_rot[:, 1], occ_rot[:, 2]
    allr = np.vstack([major_rot, minor_rot])
    xs = np.concatenate([ox, allr[:, 0]]); ys = np.concatenate([oy, allr[:, 1]])
    zb = ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), (a.buffer, a.buffer))
    if a.gpu:
        rasterize_triangles_gpu(zb, occ_tris, ox, oy, oz)
    else:
        rasterize_triangles(zb, occ_tris, ox, oy, oz, progress=not a.no_progress)

    major_segs = clip_hidden_lines(zb, major_rot, major_idx, margin=a.clip_margin)
    minor_segs = clip_hidden_lines(zb, minor_rot, minor_idx, margin=a.clip_margin)

    # Vertical hatching on the front (real-axis) wall — the scalloped zeros edge.
    front_im = np.full(80, i_min)
    front_re = np.linspace(r_min, r_max, 80)
    front_h = surface.height_at(front_re, front_im)
    hatch = [project(np.array([[i, r, 0.0], [i, r, h]]))[:, :2]
             for i, r, h in zip(front_im, front_re, front_h)]

    fig, ax = plt.subplots(figsize=(16, 11))
    for xy in major_segs:
        ax.plot(xy[:, 0], xy[:, 1], lw=0.4, c="k")
    for xy in minor_segs:
        ax.plot(xy[:, 0], xy[:, 1], lw=0.15, c="k")
    for xy in hatch:
        ax.plot(xy[:, 0], xy[:, 1], lw=0.25, c="k")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(f"{a.output_prefix}_hi_res.svg")
    plt.close(fig)
    print(f"Wrote {a.output_prefix}_hi_res.svg")


if __name__ == "__main__":
    main()
