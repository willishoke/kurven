"""End-to-end equivalence: the Swift bake against the Python plate.

The cheap tests (`tests/check_bundle.py`, `swift run kurven-test`) pin down the
schema, the array reader, the camera and the clipper. Everything they leave is
rasterization: the Python Z-buffer is filled by a CPU barycentric loop or by
moderngl, the Swift one by Metal, and two rasterizers disagree along triangle
edges no matter how carefully they are written. So this test is tolerance-based
by nature -- as Python's own CPU-vs-GPU paths already are -- and it reports the
size of the disagreement rather than asserting it away.

Two comparisons:

  depth   per-pixel |Δ| and the fraction of pixels that differ, over the same
          frame. The frames must match first; if they do not, everything below
          is measuring the framing rather than the drawing, and the script says
          so and stops.

          Which oracle matters here. Against Python's *CPU* barycentric loop --
          the definition, and the thing `ZBuffer.index_to_coord` describes --
          the Metal buffer agrees to about 1e-5 of the height span. Against
          moderngl it does not, because moderngl maps the coordinate range
          straight onto [-1, 1] and so samples half a pixel off its own
          lattice; the Swift renderer aligns to the lattice instead. At bake
          resolution that half pixel is under 1% of the picture, but at 400
          squared it is a quarter of it, which is why `--cpu` is the meaningful
          comparison and the default is only a sanity check.
  strokes per layer: path count, total ink length, and the directed Hausdorff
          distance from each Swift vertex to the nearest Python vertex (in
          plate units, and as a fraction of the plate's diagonal). Ink length
          and Hausdorff catch different failures -- a hidden-line bug changes
          the length while leaving every surviving vertex on a Python curve;
          a camera bug moves every vertex while preserving the length.

    python tests/compare_bake.py recip --res 800 --buffer 1500

Requires the Swift package to be built (`swift build --package-path KurvenSwift`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kurven.bundle import indices_from_csr  # noqa: E402
from kurven.export import export, load_example  # noqa: E402
from kurven.projection import Projection  # noqa: E402

def _cli():
    """Prefer the release binary: the depth readback and the clip are tight
    scalar loops, and a debug build spends more time bounds-checking them than
    the GPU spends drawing."""
    for config in ("release", "debug"):
        path = ROOT / "KurvenSwift" / ".build" / config / "kurven-cli"
        if path.exists():
            return path
    return ROOT / "KurvenSwift" / ".build" / "release" / "kurven-cli"


CLI = _cli()


def python_plate(module, ex_args, gpu=True):
    """Run the example's own camera-dependent half -- the plate itself."""
    scene = module.build_scene(ex_args, verbose=False)
    p = scene.preset.plate
    project = Projection(shear=p.shear, x_angle=p.x_angle, z_angle=p.z_angle,
                         flip_x=p.flip_x, y_scale=p.y_scale)
    drawn, zb = module.render_plate(scene, project, buffer=ex_args.buffer,
                                    gpu=gpu, clip_margin=scene.preset.margin,
                                    progress=False, verbose=False)
    return scene, project, drawn, zb


def swift_depth(bundle, path, resolution):
    """Render the same occluding geometry with Metal and read it back."""
    subprocess.run(
        [str(CLI), "depth", str(bundle), "--resolution", str(resolution),
         "-o", str(path)],
        check=True, capture_output=True)
    return np.load(path).astype(np.float64)


def compare_depth(zb, swift, meta):
    """Per-pixel agreement between the two rasterizers.

    Tolerance-based by nature: a barycentric CPU loop, moderngl, and Metal all
    disagree along triangle edges, and Python's own CPU and GPU paths already
    differ the same way. What matters is that the *bulk* agrees to float
    precision and the disagreement is confined to silhouettes.
    """
    if zb.buffer.shape != swift.shape:
        return None, f"shape {zb.buffer.shape} vs {swift.shape}"
    lo = np.array([zb.lower[0], zb.lower[1]])
    hi = lo + zb.image_size
    want = np.array([meta["axis0"], meta["axis1"]]).T
    if not np.allclose([lo, hi], want, atol=1e-9):
        return None, (f"frames differ: python axis0 [{lo[0]:.6f}, {hi[0]:.6f}] "
                      f"axis1 [{lo[1]:.6f}, {hi[1]:.6f}] vs swift "
                      f"axis0 {meta['axis0']} axis1 {meta['axis1']}")

    py_filled = np.isfinite(zb.buffer)
    sw_filled = np.isfinite(swift)
    both = py_filled & sw_filled
    coverage = float((py_filled ^ sw_filled).sum()) / zb.buffer.size
    if not both.any():
        return None, "the two buffers have no filled pixel in common"
    d = np.abs(zb.buffer[both] - swift[both])
    span = float(zb.buffer[both].max() - zb.buffer[both].min()) or 1.0
    return {
        "coverage_mismatch": coverage,
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p999": float(np.quantile(d, 0.999)),
        "span": span,
        "differing": float((d > 1e-3 * span).sum()) / max(both.sum(), 1),
    }, None


def swift_bake(bundle, prefix, resolution):
    subprocess.run(
        [str(CLI), "bake", str(bundle), "--resolution", str(resolution),
         "--dump", str(prefix), "-o", f"{prefix}.svg"],
        check=True, capture_output=True)
    meta = json.loads(Path(f"{prefix}.frame.json").read_text())
    out = {}
    for name in meta["layers"]:
        verts = np.load(f"{prefix}.{name}.npy").astype(float)
        offsets = np.load(f"{prefix}.{name}.idx.npy").astype(np.int64)
        out[name] = [verts[offsets[i]:offsets[i + 1]]
                     for i in range(len(offsets) - 1)]
    return meta, out


def ink_length(paths):
    total = 0.0
    for p in paths:
        p = np.asarray(p)
        if len(p) < 2:
            continue
        total += float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum())
    return total


def hausdorff(a_paths, b_paths):
    """Directed distance from every vertex of `a` to the nearest vertex of `b`.

    Vertices, not curves: a per-segment distance would be tighter, but these
    layers are densely sampled contours and the difference is far below the
    tolerance that matters here.
    """
    a = np.vstack([np.asarray(p)[:, :2] for p in a_paths]) if a_paths else np.zeros((0, 2))
    b = np.vstack([np.asarray(p)[:, :2] for p in b_paths]) if b_paths else np.zeros((0, 2))
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    d, _ = cKDTree(b).query(a)
    return float(d.max()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", choices=["recip", "elliptic", "zeta"])
    ap.add_argument("--keep", type=str, default=None,
                    help="directory for the bundle and dumps (default: a temp dir)")
    ap.add_argument("--stroke-tolerance", type=float, default=0.02,
                    help="max Hausdorff distance, as a fraction of the plate diagonal")
    ap.add_argument("--ink-tolerance", type=float, default=0.05,
                    help="max relative difference in total ink length per layer")
    ap.add_argument("--depth-tolerance", type=float, default=0.02,
                    help="max fraction of shared depth pixels that may differ")
    ap.add_argument("--cpu", action="store_true",
                    help="use Python's CPU barycentric rasterizer as the oracle "
                         "instead of moderngl. Slow, but it is the definition: "
                         "moderngl maps the coordinate range straight onto "
                         "[-1, 1], half a pixel off the lattice its own "
                         "index_to_coord describes, and the Swift renderer "
                         "matches the lattice instead.")
    args, rest = ap.parse_known_args()

    if not CLI.exists():
        raise SystemExit(f"{CLI} is not built; run "
                         f"swift build -c release --package-path KurvenSwift")

    module = load_example(args.example)
    ex_args = module.parser().parse_args(rest)
    ex_args.chunk_count = 1

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.keep) if args.keep else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)

        print(f"{args.example}: building the scene once, drawing it twice"
              f" ({'CPU' if args.cpu else 'moderngl'} oracle)")
        scene, project, drawn, zb = python_plate(module, ex_args,
                                                 gpu=not args.cpu)
        bundle = work / f"{args.example}.kurven"
        export(scene, bundle, chunk_count=1)

        resolution = ex_args.buffer
        meta, swift = swift_bake(bundle, work / args.example, resolution)

        print(f"\n  frame  {meta['shape'][0]}x{meta['shape'][1]}  "
              f"axis0 [{meta['axis0'][0]:.4f}, {meta['axis0'][1]:.4f}]  "
              f"axis1 [{meta['axis1'][0]:.4f}, {meta['axis1'][1]:.4f}]")

        allpts = np.vstack([np.asarray(s)[:, :2]
                            for _, segs in drawn for s in segs if len(s) >= 2])
        diagonal = float(np.hypot(*(allpts.max(axis=0) - allpts.min(axis=0))))
        print(f"  plate diagonal {diagonal:.4f}")

        stats, why = compare_depth(zb, swift_depth(bundle, work / f"{args.example}.depth.npy",
                                                   resolution), meta)
        if why is not None:
            print(f"  depth  NOT COMPARABLE: {why}")
            failures_depth = [f"depth: {why}"]
        else:
            print(f"  depth  max |Δ| {stats['max']:.3e}  mean {stats['mean']:.3e}  "
                  f"99.9% {stats['p999']:.3e}  (z span {stats['span']:.3f})")
            print(f"         {stats['differing']:.4%} of shared pixels differ by "
                  f">0.1% of span; {stats['coverage_mismatch']:.4%} of pixels "
                  f"are filled by one rasterizer and not the other")
            failures_depth = []
            if stats["differing"] > args.depth_tolerance:
                failures_depth.append(
                    f"depth: {stats['differing']:.2%} of shared pixels differ")
        print()

        header = (f"  {'layer':<12} {'py paths':>8} {'sw paths':>8} "
                  f"{'py ink':>10} {'sw ink':>10} {'Δink':>7} "
                  f"{'max d':>9} {'mean d':>9}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        failures = list(failures_depth)
        for layer, segments in drawn:
            name = layer.name
            sw = swift.get(name, [])
            py_ink, sw_ink = ink_length(segments), ink_length(sw)
            d_max, d_mean = hausdorff(sw, segments)
            d_ink = abs(sw_ink - py_ink) / py_ink if py_ink > 0 else 0.0
            print(f"  {name:<12} {len(segments):>8} {len(sw):>8} "
                  f"{py_ink:>10.2f} {sw_ink:>10.2f} {d_ink:>6.1%} "
                  f"{d_max:>9.5f} {d_mean:>9.5f}")
            if d_ink > args.ink_tolerance:
                failures.append(f"{name}: ink differs by {d_ink:.1%}")
            if d_max > args.stroke_tolerance * diagonal:
                failures.append(
                    f"{name}: a Swift vertex is {d_max / diagonal:.2%} of the "
                    f"diagonal from any Python vertex")

        print()
        if failures:
            for f in failures:
                print(f"  FAIL  {f}")
            return 1
        print("  within tolerance")
        return 0


if __name__ == "__main__":
    sys.exit(main())
