"""Refactor-equivalence harness (not a golden test).

The examples are nondeterministic in production because contourpy's threaded
backend stitches chunk seams in thread-completion order (chunk_count =
os.cpu_count()). That's harmless for the art but defeats exact before/after
diffing. Forcing os.cpu_count()->1 collapses contouring to a single chunk,
which is fully deterministic, so a *behaviour-preserving* refactor must produce
pixel-identical output in this mode.

Usage:
    python tests/verify_refactor.py run  <example.py> <out_prefix> [-- args...]
    python tests/verify_refactor.py diff <prefix_a> <prefix_b>

`run` executes an example's main() with os.cpu_count patched to 1 and argv set
from the trailing args. `diff` rasterizes two *_hi_res.svg outputs and reports
the max per-pixel delta and the number of differing pixels (0/0 == identical).
"""
import importlib.util
import os
import subprocess
import sys


def run_example(path, out_prefix, extra_args):
    os.cpu_count = lambda: 1  # force single-chunk -> deterministic contouring
    spec = importlib.util.spec_from_file_location("_ex", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path, "--output-prefix", out_prefix] + list(extra_args)
    spec.loader.exec_module(mod)
    mod.main()


def diff(prefix_a, prefix_b, width=1000):
    import numpy as np
    from PIL import Image

    def raster(prefix):
        svg = f"{prefix}_hi_res.svg"
        png = f"{prefix}_cmp.png"
        subprocess.run(["rsvg-convert", "-w", str(width), svg, "-o", png],
                       check=True, capture_output=True)
        return np.asarray(Image.open(png).convert("L"), dtype=float)

    a, b = raster(prefix_a), raster(prefix_b)
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH {a.shape} vs {b.shape}")
        return 1
    d = np.abs(a - b)
    n = int((d > 0).sum())
    print(f"max_pixel_delta={d.max():.0f}  differing_pixels={n}  "
          f"({'IDENTICAL' if n == 0 else 'DIFFERS'})")
    return 0 if n == 0 else 2


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        path, out_prefix = sys.argv[2], sys.argv[3]
        extra = sys.argv[4:]
        if extra and extra[0] == "--":
            extra = extra[1:]
        run_example(path, out_prefix, extra)
    elif cmd == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(1)
