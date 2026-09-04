"""`python -m kurven.export` — write an example's `Scene` as a `.kurven` bundle.

This is the whole Python side of the frontend contract. It runs an example's
`build_scene` (the camera-independent half: sample → contour → lift → boundary
geometry) and serializes the result; it never projects, rasterizes or clips,
because those are exactly the stages a realtime consumer has to do itself.

    python -m kurven.export recip -o recip.kurven --res 1600
    python -m kurven.export elliptic -o elliptic.kurven --res 2000

Determinism
-----------
contourpy's threaded backend stitches chunk seams in thread-completion order, so
a bundle written with more than one chunk is a sample from a race rather than a
value. The exporter forces one chunk unless told otherwise and records the count
in `provenance.cpuCount`, so a reader comparing two bundles can tell whether a
difference means anything.

Axis order
----------
Everything written here is in world order, `(x = real, y = imag, z = |f|)`, and
`kurven.bundle.swap_to_world` is the only place the exchange from the library's
`(imag, real, z)` happens.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np

from kurven.bundle import (
    Domain,
    GridRef,
    Interval,
    LayerSpec,
    Manifest,
    FullRegion,
    InsideRegion,
    NoWalls,
    Occluder,
    Provenance,
    SCHEMA,
    AXES,
    WallMesh,
    WallPerimeter,
    csr_from_indices,
    swap_to_world,
    write_bundle,
)
from kurven.occluder import concat_meshes

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

#: Bundle name → example module file. An example qualifies when it exposes
#: `parser()` and `build_scene(args)`; `gamma` does not yet (its camera is
#: inlined in its geometry), so it is absent by design rather than by oversight.
EXAMPLES = {
    "recip": "recip_factorial.py",
    "elliptic": "elliptic.py",
    "zeta": "zeta.py",
}


def load_example(name):
    """Import an example by bundle name. Examples are scripts, not a package, so
    they are loaded by path rather than imported."""
    if name not in EXAMPLES:
        raise SystemExit(f"unknown example {name!r}; have {sorted(EXAMPLES)}")
    path = EXAMPLES_DIR / EXAMPLES[name]
    spec = importlib.util.spec_from_file_location(f"_kurven_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for attr in ("parser", "build_scene"):
        if not hasattr(module, attr):
            raise SystemExit(f"example {name!r} has no {attr}(); not exportable yet")
    return module


def manifest_of(scene, *, phase=True, chunk_count=1, wall_mesh=True):
    """The `Manifest` describing `scene`, with the file names the bundle uses.

    Layer file names come from the layer names, so a bundle directory reads as
    the layer list. Grids are `(ny, nx)` with rows indexing imag — the transpose
    of the library's `(n_real, n_imag)` — because that is the orientation a
    consumer wants to upload as a texture whose x axis is real.
    """
    s = scene.surface
    ny, nx = len(s.imag), len(s.real)
    layers = tuple(
        LayerSpec(l.name, l.role, f"layers/{l.name}.npy", f"layers/{l.name}.idx.npy",
                  l.width, l.height_policy, l.color, l.clipped)
        for l in scene.layers)
    if wall_mesh and scene.walls:
        walls = WallMesh("occluder/walls.npy", "occluder/walls.tri.npy")
    elif scene.perimeter is not None:
        walls = WallPerimeter(scene.perimeter, scene.wall_base)
    else:
        walls = NoWalls()
    return Manifest(
        schema=SCHEMA,
        axes=AXES,
        domain=Domain(Interval(float(s.real[0]), float(s.real[-1])),
                      Interval(float(s.imag[0]), float(s.imag[-1]))),
        height=GridRef("height.npy", (ny, nx), "<f4"),
        phase=GridRef("phase.npy", (ny, nx), "<f4") if phase else None,
        caps=scene.caps,
        occluder=Occluder(scene.occluder_step, scene.tiles, walls,
                          FullRegion() if scene.region is None
                          else InsideRegion(scene.region),
                          scene.wall_base),
        layers=layers,
        presets=(scene.preset,),
        provenance=Provenance(scene.function, scene.params, chunk_count),
    )


def arrays_of(scene, *, wall_mesh=True):
    """`(height, phase, layers, walls)` in world order, ready for `write_bundle`.

    `height` is |f| **unclamped**: the cap is a separate, editable property of
    the model (`Manifest.caps`), and a consumer that clamps at upload time can
    change the cap without resampling.
    """
    s = scene.surface
    height = np.ascontiguousarray(s.mag.T, dtype=np.float32)
    phase = np.ascontiguousarray(s.angle.T, dtype=np.float32)

    layers = {}
    for l in scene.layers:
        verts, offsets = csr_from_indices(swap_to_world(l.xyz), l.indices)
        layers[l.name] = (verts, offsets)

    walls = None
    if wall_mesh and scene.walls:
        wv, wt = concat_meshes(scene.walls)
        walls = (swap_to_world(wv), wt)
    return height, phase, layers, walls


def export(scene, path, *, chunk_count=1, phase=True, wall_mesh=True):
    """Serialize a `Scene` to `path` and return the `Manifest` written."""
    manifest = manifest_of(scene, phase=phase, chunk_count=chunk_count,
                           wall_mesh=wall_mesh)
    height, ph, layers, walls = arrays_of(scene, wall_mesh=wall_mesh)
    write_bundle(path, manifest=manifest, height=height,
                 phase=ph if phase else None, layers=layers, walls=walls)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m kurven.export",
        description="Write an example's camera-independent scene as a .kurven bundle.")
    ap.add_argument("example", choices=sorted(EXAMPLES))
    ap.add_argument("-o", "--output", default=None,
                    help="bundle directory (default <example>.kurven)")
    ap.add_argument("--chunk-count", type=int, default=1,
                    help="contourpy chunks; 1 (the default) is reproducible")
    ap.add_argument("--no-phase", action="store_true",
                    help="omit phase.npy (halves the bundle; blocks phase-colored modes)")
    ap.add_argument("--derived", action="store_true",
                    help="describe the walls as a perimeter instead of dumping "
                         "their mesh; the consumer rebuilds them from height.npy")
    ap.add_argument("--quiet", action="store_true")
    args, rest = ap.parse_known_args(argv)

    module = load_example(args.example)
    ex_args = module.parser().parse_args(rest)
    ex_args.chunk_count = args.chunk_count

    scene = module.build_scene(ex_args, verbose=not args.quiet)
    if args.derived:
        if scene.perimeter is None:
            raise SystemExit(f"{args.example}: no perimeter to derive walls from")
        scene = dataclasses.replace(scene, walls=())

    out = Path(args.output or f"{args.example}.kurven")
    manifest = export(scene, out, chunk_count=args.chunk_count,
                      phase=not args.no_phase, wall_mesh=not args.derived)

    if not args.quiet:
        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"wrote {out} ({total / 1e6:.1f} MB)")
        for spec in manifest.layers:
            n = len(np.load(out / spec.offsets)) - 1
            print(f"  {spec.name:<12} {n:>7} paths  lw={spec.width}"
                  f"{'' if spec.clipped else '  (unclipped)'}")


if __name__ == "__main__":
    main()
