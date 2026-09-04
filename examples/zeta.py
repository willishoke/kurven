"""Riemann zeta analytic landscape.

Ported from 2401/zeta.ipynb, with one deliberate departure: the notebook built
its Z-buffer by Delaunay-triangulating the union of contour vertices and
boundary curves, which produced a cutout-shaped buffer only as a side effect of
the contour filter (no vertices in the notch, so no triangles there). That is a
notebook trick, not a surface. Here the occluder is what it is in every other
example — the clamped |ζ| heightfield, meshed over the staircase region and
skirted with cut-face wall curtains — and the staircase is a single
`Perimeter` from which the mesh mask, the walls, the ground ink and the hatch
all derive.

Two visible consequences, both accepted:

  - The occluder now tracks the surface everywhere inside the staircase rather
    than only where contour vertices happened to fall, so hidden-line removal
    is correct in the vertex-free plateaus instead of accidentally permissive.
  - The notebook's `c1|c2|c3|c4` keep-predicate disagreed with the polygon it
    was meant to describe by one unit in imag (it kept `im > -15` where the
    drawn ground line sits at `im = -14`). The polygon wins; a sliver of
    contour beyond the drawn edge is gone.

`Projection.z_clamp` also goes away: the cap is a property of the model, not
the camera, so heights come from `Surface.height_at` (clamped) and the
projection is pure camera.

Run:
    python examples/zeta.py --gpu
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from kurven.bench import PhaseTimer
from kurven.bundle import Affine2, CameraPreset, PlateProjection, UniformCap
from kurven.contours import contour_levels
from kurven.occluder import build_occluder
from kurven.outline import clip_hidden_lines
from kurven.perimeter import Edge, Perimeter
from kurven.projection import Projection
from kurven.scene import InkLayer, Scene
from kurven.surface import Surface
from kurven.zbuffer import (
    ZBuffer,
    rasterize_triangles,
    rasterize_triangles_gpu,
)


DEFAULT_CACHE = "/Users/willishoke/journal/2401/zeta_5000.npy"

R_MIN, R_MAX = -6.0, 8.0
I_MIN, I_MAX = -30.0, 30.0
Z_LIMIT = 6.0

#: The staircase cutout, closed, in `(imag, real)`. Edges 0-7 are the drawn
#: ground outline; 8 and 9 close the loop along the far and right boundaries of
#: the sampled rectangle, where there is a cut face but no ink.
CUTOUT_CORNERS = [
    (28.0, R_MIN), (0.0, R_MIN), (0.0, -2.0), (-5.0, -2.0), (-5.0, -0.5),
    (-14.0, -0.5), (-14.0, 0.5), (-28.0, 0.5), (-28.0, R_MAX), (28.0, R_MAX),
]
GROUND_EDGES = 8          # of the 10, the ones drawn at z = 0

#: Hatch strokes per unit length along a cut face, and which edges get them
#: (the notebook hatches every staircase edge but the short re-axis one at
#: im = 0).
HATCH_PER_UNIT = 6
HATCH_EDGES = (0, 2, 4, 6, 3, 5, 7)
#: Wall-curtain samples per unit length; the occluder wants more than the ink.
WALL_PER_UNIT = 20


def cutout():
    """The staircase `Perimeter`. One definition; four consumers."""
    n = len(CUTOUT_CORNERS)
    return Perimeter([Edge(CUTOUT_CORNERS[i], CUTOUT_CORNERS[(i + 1) % n])
                      for i in range(n)])


def _edge_lengths(perim):
    return [float(np.hypot(e.end[0] - e.start[0], e.end[1] - e.start[1]))
            for e in perim.edges]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=DEFAULT_CACHE)
    p.add_argument("--buffer", type=int, default=8000)
    p.add_argument("--output-prefix", type=str, default="zeta")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--clip-margin", type=float, default=0.2,
                   help="Matches the notebook's cell 22 margin.")
    p.add_argument("--occluder-res", type=int, default=2500,
                   help="Target resolution along the cache's long axis for the "
                        "heightfield occluder mesh.")
    p.add_argument("--chunk-count", type=int, default=None,
                   help="contourpy chunks; 1 makes contouring deterministic")
    return p


def build_scene(a, *, verbose=True, timer=None):
    """The camera-independent half: load the cached ζ grid, contour it, lift the
    contours, and build the staircase boundary and its ink."""
    tick = timer.tick if timer is not None else (lambda _: None)

    tick("load cache")
    comp = np.load(a.cache)
    surface = Surface.from_cache(comp, (R_MIN, R_MAX), (I_MIN, I_MAX),
                                 z_limit=Z_LIMIT)
    mag, angle = surface.mag, surface.angle
    if verbose:
        print(f"      cached grid: {comp.shape}, |zeta| ∈ "
              f"[{mag.min():.3f}, {mag.max():.3f}]")

    perim = cutout()
    inside = lambda xyz: perim.contains(xyz[:, 0], xyz[:, 1])

    # Contour levels exactly per the notebook
    mag_levels = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0,
                           3.0, 4.0, 5.0, 6.0])
    angle_levels = np.linspace(-np.pi, np.pi, 11)
    peak_levels = [2.5, 3.5, 4.5, 5.5]

    tick("contour generation")
    cc = a.chunk_count
    rb, ib = (R_MIN, R_MAX), (I_MIN, I_MAX)
    mag_paths = contour_levels(mag, mag_levels, rb, ib, chunk_count=cc)
    angle_paths = contour_levels(angle, angle_levels, rb, ib, chunk_count=cc)
    peak_paths = contour_levels(mag, peak_levels, rb, ib, chunk_count=cc)

    tick("lift contours")
    # Magnitude isocontours sit at their level; angle contours take the
    # unclamped surface magnitude and drop vertices above the cap. All are
    # restricted to the staircase. Peak contours are additionally restricted to
    # |imag| < 5, which is what focuses detail on the s = 1 pole.
    mag_xyz, mag_idx, p = surface.lift_contours(
        mag_paths, start=0, height="level", keep=inside)
    ang_xyz, ang_idx, p = surface.lift_contours(
        angle_paths, start=p, height="magnitude",
        keep=lambda xyz: (xyz[:, 2] <= Z_LIMIT) & inside(xyz))
    peak_xyz, peak_idx, p = surface.lift_contours(
        peak_paths, start=p, height="level",
        keep=lambda xyz: inside(xyz) & (xyz[:, 0] < 5) & (xyz[:, 0] > -5))

    tick("rim curves")
    # The crest of each cut face: sample the surface along the staircase so the
    # top edge of every wall is drawn. In the notebook these existed to give the
    # Delaunay buffer coverage at the polygon edges; with a heightfield occluder
    # they are purely ink, and they are exactly the perimeter at surface height.
    eps = 0.01
    top_rear_cutoff = -2.0
    while surface.mag_at(top_rear_cutoff, 28.0) > Z_LIMIT:
        top_rear_cutoff += eps
    top_rear_cutoff -= eps
    top_front_cutoff = 10.0
    while surface.mag_at(R_MIN, top_front_cutoff) > Z_LIMIT:
        top_front_cutoff -= eps
    top_front_cutoff += eps

    def rim(im_arr, re_arr):
        return np.column_stack([im_arr, re_arr,
                                surface.height_at(re_arr, im_arr)])

    rim_3d = [
        rim(np.linspace(0, top_front_cutoff, 100), np.full(100, R_MIN)),
        rim(np.linspace(0, -5, 100), np.full(100, -2.0)),
        rim(np.full(100, -5.0), np.linspace(-2, -0.5, 100)),
        rim(np.linspace(-5, -14, 100), np.full(100, -0.5)),
        rim(np.full(100, -14.0), np.linspace(0.5, -0.5, 100)),
        rim(np.linspace(-14, -28, 100), np.full(100, 0.5)),
        rim(np.full(100, -28.0), np.linspace(0.5, R_MAX, 100)),
        rim(np.linspace(-28, 28, 1000), np.full(1000, R_MAX)),
        rim(np.full(100, 28.0), np.linspace(top_rear_cutoff, R_MAX, 100)),
    ]

    tick("boundary geometry")
    # The staircase at z = 0, plus the corner posts that stand the cut faces up
    # and the two radius lines across the pole cap. The ground lines come from
    # the perimeter; only the posts are bespoke.
    posts_3d = [
        np.array([[28., R_MIN, Z_LIMIT], [28., R_MIN, 0.]]),
        np.array([[28., R_MIN, Z_LIMIT], [28., top_rear_cutoff, Z_LIMIT]]),
        np.array([[28., R_MIN, Z_LIMIT], [top_front_cutoff, R_MIN, Z_LIMIT]]),
    ] + [
        np.array([[im, re, 0.], [im, re, float(surface.height_at(re, im))]])
        for im, re in [(-5., -2.), (-5., -0.5), (-14., -0.5), (-14., 0.5),
                       (-28., 0.5), (-28., R_MAX)]
    ]
    ground_3d = perim.ground_polygon_3d()[:GROUND_EDGES] + posts_3d

    lengths = _edge_lengths(perim)
    hatch_3d = []
    for k in HATCH_EDGES:
        n = max(3, int(lengths[k] * HATCH_PER_UNIT))
        hatch_3d += perim.edges[k].wall_hatch_3d(surface, n, trim=True)

    # The pole cap at z = Z_LIMIT: horizontal strokes from the front wall in to
    # wherever |ζ| drops back under the cap.
    caps_3d = []
    n_cap = max(2, int((28 - top_front_cutoff) * HATCH_PER_UNIT // 2))
    for im in np.linspace(28, top_front_cutoff, n_cap)[1:-1]:
        edge = R_MIN
        while surface.mag_at(edge, im) > Z_LIMIT:
            edge += eps
        caps_3d.append(np.array([[im, R_MIN, Z_LIMIT], [im, edge - eps, Z_LIMIT]]))

    if verbose:
        print(f"      {len(ground_3d)} polygon lines, {len(hatch_3d)} vertical, "
              f"{len(caps_3d)} top")

    layers = (
        InkLayer("mag", "magnitude", mag_xyz, _strata(mag_xyz, mag_idx), 0.4,
                 height_policy="level"),
        InkLayer("ang", "phase", ang_xyz, _strata(ang_xyz, ang_idx), 0.4,
                 height_policy="magnitude"),
        InkLayer("peak", "magnitude", peak_xyz, _strata(peak_xyz, peak_idx), 0.4,
                 height_policy="level"),
        InkLayer.from_segments("rim", "scaffold", rim_3d, 0.4),
        InkLayer.from_segments("ground", "scaffold", ground_3d, 0.3, clipped=False),
        InkLayer.from_segments("wall_hatch", "scaffold", hatch_3d, 0.3, clipped=False),
        InkLayer.from_segments("cap_hatch", "scaffold", caps_3d, 0.3, clipped=False),
    )

    wall_density = [max(3, int(l * WALL_PER_UNIT)) for l in lengths]
    return Scene(
        function="zeta",
        params={"cache": a.cache, "rMin": R_MIN, "rMax": R_MAX, "iMin": I_MIN,
                "iMax": I_MAX, "zLimit": Z_LIMIT,
                "nReal": int(comp.shape[0]), "nImag": int(comp.shape[1])},
        surface=surface,
        layers=layers,
        preset=CameraPreset(
            "zeta", PlateProjection(-0.18, -79.5, -90.0, True, 0.75),
            a.clip_margin, a.buffer),
        caps=UniformCap(Z_LIMIT),
        occluder_step=max(1, max(comp.shape) // a.occluder_res),
        tiles=(Affine2.identity(),),
        perimeter=perim.to_world(wall_density),
        walls=tuple(perim.wall_curtains(surface, wall_density)),
    )


#: Imag values at which a contour path is cut into separate strokes. A ζ level
#: set can wander the full 60 units of imag; without the cuts one path welds
#: into a single stroke that jumps the cutout.
STRATA = (14.0, 5.0, 0.0)
_BUMP = 1_000_000


def _strata(xyz, indices):
    """Split path tags by imag stratum (the notebook's BUMP arithmetic)."""
    out = np.asarray(indices).copy()
    for s in STRATA:
        out[xyz[:, 0] < s] += _BUMP
    return out


def render_plate(scene, project, *, buffer, gpu=False, clip_margin=0.2,
                 progress=True, verbose=True, timer=None):
    """The camera-dependent half: project, rasterize the occluder, clip."""
    tick = timer.tick if timer is not None else (lambda _: None)
    perim = cutout()

    tick("build occluder mesh")
    occ_verts, occ_tris = build_occluder(
        scene.surface, scene.occluder_step, walls=scene.walls,
        keep=lambda im, re: perim.contains(im, re))
    if verbose:
        print(f"      occluder: {len(occ_verts)} verts, {len(occ_tris)} tris")

    tick("project")
    occ_rot = project(occ_verts)
    ox, oy, oz = occ_rot[:, 0], occ_rot[:, 1], occ_rot[:, 2]
    rotated = [project(l.xyz) for l in scene.layers]

    tick("rasterize")
    clipped = [r for l, r in zip(scene.layers, rotated) if l.clipped]
    allr = np.vstack(clipped) if clipped else np.zeros((0, 3))
    xs = np.concatenate([ox, allr[:, 0]])
    ys = np.concatenate([oy, allr[:, 1]])
    zb = ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), (buffer, buffer))
    if gpu:
        rasterize_triangles_gpu(zb, occ_tris, ox, oy, oz)
    else:
        rasterize_triangles(zb, occ_tris, ox, oy, oz, progress=progress)

    tick("clip")
    out = []
    for layer, rot in zip(scene.layers, rotated):
        if layer.clipped:
            out.append((layer, clip_hidden_lines(zb, rot, layer.indices,
                                                 margin=clip_margin)))
        else:
            out.append((layer, layer.split(rot[:, :2])))
    return out


def main():
    args = parser().parse_args()
    if args.backend:
        matplotlib.use(args.backend)
    timer = PhaseTimer()

    scene = build_scene(args, timer=timer)
    project = Projection(shear=-0.18, x_angle=-79.5, z_angle=-90, flip_x=True,
                         y_scale=0.75)
    drawn = render_plate(scene, project, buffer=args.buffer, gpu=args.gpu,
                         clip_margin=args.clip_margin,
                         progress=not args.no_progress, timer=timer)

    timer.tick("save hi_res.svg")
    fig, ax = plt.subplots(figsize=(16, 16))
    for layer, segments in drawn:
        for xy in segments:
            ax.plot(xy[:, 0], xy[:, 1], lw=layer.width, c=layer.color)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(f"{args.output_prefix}_hi_res.svg")
    plt.close(fig)
    timer.done()
    print(f"Wrote {args.output_prefix}_hi_res.svg.")


if __name__ == "__main__":
    main()
