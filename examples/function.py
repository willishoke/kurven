"""Any function's analytic landscape — the plate that is not written in advance.

The other four examples each reproduce one published plate, and each one knows
its function, its window, its truncation and its hatching because a person chose
them. This one is handed all four on the command line and derives the rest
(`kurven.landscape`). It exists for three reasons:

  - it is the Python half of the frontend's function picker, reachable through
    `kurven.serve`'s `landscape` method, which is the same code by another door;
  - it is the oracle. `tests/compare_bake.py function` draws the plate in Python
    and the bundle in Swift and compares them, which is how the hatching
    derived from a description in Swift is held to the hatching computed from
    the analytic function here;
  - it is the way to get a landscape of your own function as an SVG without
    opening the app.

Run:
    python examples/function.py --function gamma --gpu
    python examples/function.py --expression "exp(1/z)" --r-min -1 --r-max 1 \\
        --i-min -1 --i-max 1 --cap 4
    python -m kurven.export function -o mine.kurven --derived \\
        --expression "zeta(z)" --res 800
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from kurven.bundle import Domain, Interval, NoCaps, RealBand, RealBandCaps, UniformCap
from kurven.landscape import CATALOG, DEFAULT_RESOLUTION, Spec, preset
from kurven.occluder import build_occluder
from kurven.outline import clip_hidden_lines
from kurven.projection import Projection
from kurven.zbuffer import ZBuffer, rasterize_triangles, rasterize_triangles_gpu
from kurven import landscape


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--function", choices=[p.name for p in CATALOG],
                    default="rgamma",
                    help="a preset from the catalog: its expression, window "
                         "and truncation")
    ap.add_argument("--expression", default=None,
                    help="f(z) in kurven's expression language, overriding "
                         "--function's (e.g. \"1/gamma(z)\", \"exp(1/z)\")")
    ap.add_argument("--r-min", type=float, default=None)
    ap.add_argument("--r-max", type=float, default=None)
    ap.add_argument("--i-min", type=float, default=None)
    ap.add_argument("--i-max", type=float, default=None)
    ap.add_argument("--res", type=int, default=DEFAULT_RESOLUTION,
                    help="samples along the longer side of the window")
    ap.add_argument("--cap", type=float, default=None,
                    help="truncate the surface at this |f|; the preset's own "
                         "cap by default")
    ap.add_argument("--no-cap", action="store_true",
                    help="do not truncate, whatever the preset says")
    ap.add_argument("--bands", default=None,
                    help="truncate per band of Re, as 'x<cap,x<cap,...,cap' "
                         "(gamma's per-spire form): "
                         "\"-3.5<3.1,-2.5<4.2,-1.5<5,0.5<5.6,5\"")
    ap.add_argument("--hatch-spacing", type=float, default=None,
                    help="world units between hatch strokes; derived from the "
                         "window by default")
    ap.add_argument("--buffer", type=int, default=4000)
    ap.add_argument("--occluder-res", type=int, default=800)
    ap.add_argument("--clip-margin", type=float, default=0.02)
    ap.add_argument("--output-prefix", default="function")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--chunk-count", type=int, default=None,
                    help="contourpy chunks; 1 makes contouring deterministic")
    return ap


def parse_bands(text):
    """`"-3.5<3.1,-2.5<4.2,5"` -> `RealBandCaps`: a staircase in Re, and the
    cap beyond the last threshold."""
    bands, beyond = [], float("inf")
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "<" in piece:
            below, cap = piece.split("<", 1)
            bands.append(RealBand(float(below), float(cap)))
        else:
            beyond = float(piece)
    return RealBandCaps(tuple(bands), beyond)


def spec_of(a):
    """The `Spec` the arguments describe: the preset, then the overrides."""
    p = preset(a.function)
    domain = Domain(
        Interval(p.domain.real.lo if a.r_min is None else a.r_min,
                 p.domain.real.hi if a.r_max is None else a.r_max),
        Interval(p.domain.imag.lo if a.i_min is None else a.i_min,
                 p.domain.imag.hi if a.i_max is None else a.i_max))
    if a.no_cap:
        caps = NoCaps()
    elif a.bands:
        caps = parse_bands(a.bands)
    elif a.cap is not None:
        caps = UniformCap(a.cap)
    elif a.expression is not None:
        caps = None                      # a function of its own: read the samples
    else:
        caps = None if p.cap is None else UniformCap(p.cap)
    return Spec(expression=a.expression or p.expression, domain=domain,
                resolution=a.res, caps=caps, spacing=a.hatch_spacing,
                name="" if a.expression else p.name)


def build_scene(a, *, verbose=True):
    """The camera-independent half. `kurven.landscape` does all of it; this is
    the argparse-shaped door into it."""
    return landscape.build_scene(spec_of(a), verbose=verbose,
                                 chunk_count=a.chunk_count or 1)


def render_plate(scene, project, *, buffer, gpu=False, clip_margin=0.02,
                 progress=True, verbose=True):
    """The camera-dependent half: project, rasterize the truncated heightfield
    into a Z-buffer, clip the ink against it.

    The same function `examples/recip_factorial.py` has, and deliberately so:
    once the caps are a property of the `Surface`, a landscape with four bands
    of truncation rasterizes exactly like one with a single limit, and there is
    no second code path for the general case to drift into.
    """
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
    p = scene.preset.plate
    project = Projection(shear=p.shear, x_angle=p.x_angle, z_angle=p.z_angle,
                         flip_x=p.flip_x, y_scale=p.y_scale)
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
