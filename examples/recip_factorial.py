"""Reciprocal gamma analytic landscape — 1/Γ(z).

A fresh example built on the kurven `Surface` noun + the grid-mesh occluder,
not ported from a notebook. Reproduces the classic Jahnke-Emde plate "Relief
der reziproken Fakultät" (Fig. 6): 1/Γ is entire with zeros at z = 0, -1, -2,
...; |1/Γ| ripples along the negative real axis (the zeros), grows without
bound toward the back-left (clamped to a flat cap), and decays toward +Re.

The example is split at the camera: `build_scene` does the expensive,
camera-independent half (sample → contour → lift → boundary geometry) and
returns a `kurven.scene.Scene`; `render_plate` takes that scene and a
`Projection` and does the camera-dependent half (project → rasterize → clip).
`main` is their composition, so the plate is unchanged; `kurven.export` calls
`build_scene` alone and writes a `.kurven` bundle.

Run:
    python examples/recip_factorial.py --gpu
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss

from kurven.bundle import (Affine2, CameraPreset, KeepAll, LayerContour,
                           PlateProjection, UniformCap)
from kurven.contours import contour_levels
from kurven.occluder import build_occluder
from kurven.outline import clip_hidden_lines
from kurven.perimeter import Perimeter
from kurven.projection import Projection
from kurven.scene import InkLayer, Scene
from kurven.surface import Surface
from kurven.zbuffer import ZBuffer, rasterize_triangles, rasterize_triangles_gpu

# Wall-curtain sample density and hatch stroke count on the front edge.
WALL_DENSITY = 600
HATCH_DENSITY = 80


def parser():
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
    ap.add_argument("--chunk-count", type=int, default=None,
                    help="contourpy chunks; 1 makes contouring deterministic")
    return ap


def build_scene(a, *, verbose=True):
    """The camera-independent half: sample 1/Γ, contour it, lift the contours
    onto the surface, and describe the occluder's boundary. Nothing here knows
    where the camera is."""
    r_min, r_max = a.r_min, a.r_max
    i_min, i_max = 0.0, a.i_max
    z_limit = a.z_limit
    res = a.res
    real = np.linspace(r_min, r_max, res)
    imag = np.linspace(i_min, i_max, res)
    rb, ib = (r_min, r_max), (i_min, i_max)

    surface = Surface.from_function(ss.rgamma, real, imag, z_limit=z_limit)
    mag, angle = surface.mag, surface.angle
    if verbose:
        print(f"|1/Γ| ∈ [{mag.min():.3f}, {mag.max():.1f}], clamp {z_limit}")

    mag_major = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    mag_minor = np.setdiff1d(np.round(np.arange(0.1, z_limit, 0.1), 2), mag_major)
    ang_major = np.linspace(-np.pi, np.pi, 5)                    # every 90 deg
    ang_minor = np.setdiff1d(np.linspace(-np.pi, np.pi, 13), ang_major)  # 30 deg

    cc = a.chunk_count
    p = 0
    Mm, Mmi, p = surface.lift_contours(
        contour_levels(mag, mag_major, rb, ib, chunk_count=cc), start=p)
    mm, mmi, p = surface.lift_contours(
        contour_levels(mag, mag_minor, rb, ib, chunk_count=cc), start=p)
    Am, Ami, p = surface.lift_contours(
        contour_levels(angle, ang_major, rb, ib, chunk_count=cc), start=p)
    am, ami, p = surface.lift_contours(
        contour_levels(angle, ang_minor, rb, ib, chunk_count=cc), start=p)

    # Occluder: clamped heightfield grid + cut-face wall curtains on all four
    # domain edges. No cutout/mask — the displayed region is the full rectangle;
    # the back-left cap falls out of the clamp.
    perim = Perimeter.rectangle((i_min, i_max), (r_min, r_max))

    # Vertical hatching on the front (real-axis) wall — the scalloped zeros edge.
    # The front edge (index 0) of the same perimeter that builds the occluder.
    hatch = perim.edges[0].wall_hatch_3d(surface, HATCH_DENSITY)

    # Layer order is draw order: both major families, then both minor, then the
    # scaffold, exactly as the pre-split `main` stacked them.
    # Every contour family here is exactly "these levels of that field, lifted
    # onto the surface" -- nothing is trimmed, tiled or snapped -- so all four
    # can be described rather than dumped, and `--derived` writes no layers at
    # all. The hatch cannot: it is geometry along a perimeter, not a level set.
    def contours(field, levels):
        return LayerContour(field, tuple(float(v) for v in levels), KeepAll())

    layers = (
        InkLayer("mag_major", "magnitude", Mm, Mmi, 0.4,
                 source=contours("magnitude", mag_major)),
        InkLayer("ang_major", "phase", Am, Ami, 0.4,
                 source=contours("phase", ang_major)),
        InkLayer("mag_minor", "magnitude", mm, mmi, 0.15,
                 source=contours("magnitude", mag_minor)),
        InkLayer("ang_minor", "phase", am, ami, 0.15,
                 source=contours("phase", ang_minor)),
        InkLayer.from_segments("wall_hatch", "scaffold", hatch, 0.25, clipped=False),
    )

    return Scene(
        function="rgamma",
        params={"res": res, "rMin": r_min, "rMax": r_max, "iMin": i_min,
                "iMax": i_max, "zLimit": z_limit},
        surface=surface,
        layers=layers,
        preset=CameraPreset(
            "recip",
            PlateProjection(a.scale, a.x_angle, a.z_angle, True, None),
            a.clip_margin, a.buffer),
        caps=UniformCap(z_limit),
        occluder_step=max(1, res // a.occluder_res),
        tiles=(Affine2.identity(),),
        perimeter=perim.to_world(WALL_DENSITY),
        walls=tuple(perim.wall_curtains(surface, WALL_DENSITY)),
    )


def render_plate(scene, project, *, buffer, gpu=False, clip_margin=0.02,
                 progress=True, verbose=True):
    """The camera-dependent half: project the scene, rasterize the occluder into
    a Z-buffer, and clip the ink against it.

    Returns `(drawn, zb)`: `drawn` is `(layer, segments)` pairs in draw order,
    `zb` the Z-buffer they were clipped against. The buffer is an output, not a
    scratch value -- the silhouette comes off it, and so does any comparison
    against another renderer's depth (`tests/compare_bake.py`)."""
    occ_verts, occ_tris = build_occluder(
        scene.surface, scene.occluder_step, walls=scene.walls)
    if verbose:
        print(f"occluder: {len(occ_verts)} verts, {len(occ_tris)} tris")

    occ_rot = project(occ_verts)
    ox, oy, oz = occ_rot[:, 0], occ_rot[:, 1], occ_rot[:, 2]
    rotated = [project(l.xyz) for l in scene.layers]

    clipped = [r for l, r in zip(scene.layers, rotated) if l.clipped]
    allr = np.vstack(clipped) if clipped else np.zeros((0, 3))
    xs = np.concatenate([ox, allr[:, 0]])
    ys = np.concatenate([oy, allr[:, 1]])
    zb = ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), (buffer, buffer))
    if gpu:
        rasterize_triangles_gpu(zb, occ_tris, ox, oy, oz)
    else:
        rasterize_triangles(zb, occ_tris, ox, oy, oz, progress=progress)

    out = []
    for layer, rot in zip(scene.layers, rotated):
        if layer.clipped:
            out.append((layer, clip_hidden_lines(zb, rot, layer.indices,
                                                 margin=clip_margin)))
        else:
            out.append((layer, layer.split(rot[:, :2])))
    return out, zb


def main():
    a = parser().parse_args()
    if not a.gpu:
        matplotlib.use("Agg")

    scene = build_scene(a)
    project = Projection(shear=a.scale, x_angle=a.x_angle, z_angle=a.z_angle,
                         flip_x=True)
    drawn, _ = render_plate(scene, project, buffer=a.buffer, gpu=a.gpu,
                            clip_margin=a.clip_margin, progress=not a.no_progress)

    fig, ax = plt.subplots(figsize=(16, 11))
    for layer, segments in drawn:
        for xy in segments:
            ax.plot(xy[:, 0], xy[:, 1], lw=layer.width, c=layer.color)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(f"{a.output_prefix}_hi_res.svg")
    plt.close(fig)
    print(f"Wrote {a.output_prefix}_hi_res.svg")


if __name__ == "__main__":
    main()
