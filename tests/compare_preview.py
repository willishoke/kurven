"""The preview against the plate, as images.

`tests/compare_bake.py` checks the Swift *bake* against the Python plate on
geometry -- path counts, ink length, Hausdorff. That leaves the preview
unchecked, and the preview is where the design's central claim lives: a GPU
depth test *is* hidden-line removal, so the realtime picture and the exact bake
are the same computation at different resolutions. The difference between them
is meant to be only that the preview tests visibility per fragment and the bake
per vertex, which can disagree on runs shorter than a pixel.

That claim is about pixels, so this compares pixels. `kurven-cli preview` writes
a PNG and the view-space rectangle it covers; this draws the Python plate's own
segments into exactly that rectangle at exactly that size, and reports how much
ink each has that the other does not, within a one-pixel tolerance.

    python tests/compare_preview.py recip

A perfect score is not the goal and would be suspicious: the two rasterize
lines differently (Metal's 1 px lines against matplotlib's stroked paths), so
some disagreement at the edges of every stroke is expected. What would matter
is ink in one picture that is nowhere near the other -- a whole contour visible
in the preview that the plate hides, or the reverse.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kurven.export import export, load_example  # noqa: E402
from kurven.projection import Projection  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from compare_bake import CLI, python_plate  # noqa: E402


def plate_raster(drawn, meta, path):
    """Draw the Python plate's segments into the preview's exact rectangle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = meta["width"], meta["height"]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(meta["viewX"][0], meta["viewX"][1])
    # viewY is (top, bottom) in screen order, which is what set_ylim wants
    # reversed: matplotlib's y grows upward.
    ax.set_ylim(meta["viewY"][1], meta["viewY"][0])
    ax.axis("off")
    ax.set_facecolor("white")
    for layer, segments in drawn:
        for xy in segments:
            ax.plot(xy[:, 0], xy[:, 1], lw=0.6, c="k", solid_capstyle="butt")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def ink(path, threshold=128):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    return a < threshold


def dilate(mask):
    """One-pixel dilation: what counts as "near" this ink."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", choices=["recip", "elliptic", "zeta", "gamma"])
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--keep", type=str, default=None)
    ap.add_argument("--tolerance", type=float, default=0.06,
                    help="max fraction of one picture's ink that may be nowhere "
                         "near the other's")
    args, rest = ap.parse_known_args()

    module = load_example(args.example)
    ex_args = module.parser().parse_args(rest)
    ex_args.chunk_count = 1

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.keep) if args.keep else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)

        print(f"{args.example}: one scene, two renderers, one rectangle")
        scene, project, drawn, _ = python_plate(module, ex_args)
        bundle = work / f"{args.example}.kurven"
        export(scene, bundle, chunk_count=1)

        preview = work / f"{args.example}.preview.png"
        subprocess.run([str(CLI), "preview", str(bundle),
                        "--width", str(args.width), "--height", str(args.height),
                        "-o", str(preview)], check=True, capture_output=True)
        meta = json.loads(preview.with_suffix(".frame.json").read_text())

        plate = work / f"{args.example}.plate.png"
        plate_raster(drawn, meta, plate)

        a, b = ink(preview), ink(plate)
        near_a, near_b = dilate(a), dilate(b)
        only_preview = (a & ~near_b).sum() / max(a.sum(), 1)
        only_plate = (b & ~near_a).sum() / max(b.sum(), 1)

        print(f"\n  {meta['width']}x{meta['height']}  "
              f"view x [{meta['viewX'][0]:.4f}, {meta['viewX'][1]:.4f}]  "
              f"view y [{meta['viewY'][1]:.4f}, {meta['viewY'][0]:.4f}]")
        print(f"  ink        preview {a.sum():>8}   plate {b.sum():>8}  "
              f"({a.sum() / b.sum():.3f}x)")
        print(f"  unmatched  preview {only_preview:>7.2%}   plate {only_plate:>7.2%}"
              "   (ink with none of the other's within a pixel)")
        print()

        worst = max(only_preview, only_plate)
        if worst > args.tolerance:
            print(f"  FAIL  {worst:.2%} of one picture's ink is nowhere near the other's")
            return 1
        print("  within tolerance")
        return 0


if __name__ == "__main__":
    sys.exit(main())
