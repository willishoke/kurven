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
    LayerFile,
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
#: `parser()` and `build_scene(args)`.
#:
#: gamma exports the uniform-sampling form of itself: a bundle carries one
#: grid, and the published plate probes for high-gradient zones and re-samples
#: those finer. `examples/gamma.py: SCENE_CAVEATS` lists what else about that
#: plate stays a Python-only bake.
EXAMPLES = {
    "recip": "recip_factorial.py",
    "elliptic": "elliptic.py",
    "zeta": "zeta.py",
    "gamma": "gamma.py",
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


def manifest_of(scene, *, phase=True, chunk_count=1, wall_mesh=True, derived=False,
                example=""):
    """The `Manifest` describing `scene`, with the file names the bundle uses.

    Layer file names come from the layer names, so a bundle directory reads as
    the layer list. Grids are `(ny, nx)` with rows indexing imag — the transpose
    of the library's `(n_real, n_imag)` — because that is the orientation a
    consumer wants to upload as a texture whose x axis is real.
    """
    s = scene.surface
    ny, nx = len(s.imag), len(s.real)
    layers = tuple(
        LayerSpec(l.name, l.role,
                  l.source if (derived and l.source is not None)
                  else LayerFile(f"layers/{l.name}.npy", f"layers/{l.name}.idx.npy"),
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
        provenance=Provenance(scene.function, scene.params, chunk_count, example),
    )


def quantize(array):
    """To float32, without claiming values the float64 original did not have.

    Rounding to float32 can round *outward*: `np.angle` returns arg in
    (-pi, pi], but float32(pi) is larger than pi, so a phase grid stored naively
    contains values above pi. A consumer contouring that grid at the level pi --
    which is what `np.linspace(-pi, pi, 5)` asks for -- then finds crossings
    where the plate found none, and draws three curves along the branch cut that
    are not in the picture.

    Clamping the quantized array into the original's range costs one pass and
    removes a whole class of "the derived layer has ink the dumped one doesn't".
    """
    array = np.ascontiguousarray(array)
    out = array.astype(np.float32)
    lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
    # The nearest float32 that is still inside [lo, hi].
    lo32 = np.float32(lo)
    if float(lo32) < lo:
        lo32 = np.nextafter(lo32, np.float32(np.inf))
    hi32 = np.float32(hi)
    if float(hi32) > hi:
        hi32 = np.nextafter(hi32, np.float32(-np.inf))
    return np.clip(out, lo32, hi32)


def arrays_of(scene, manifest, *, wall_mesh=True):
    """`(height, phase, layers, walls)` in world order, ready for `write_bundle`.

    `height` is |f| **unclamped**: the cap is a separate, editable property of
    the model (`Manifest.caps`), and a consumer that clamps at upload time can
    change the cap without resampling.
    """
    s = scene.surface
    height = quantize(s.mag.T)
    phase = quantize(s.angle.T)

    # Only the dumped layers; a described one is a statement, not an array.
    dumped = {spec.name for spec in manifest.layers if spec.files is not None}
    layers = {}
    for l in scene.layers:
        if l.name not in dumped:
            continue
        verts, offsets = csr_from_indices(swap_to_world(l.xyz), l.indices)
        layers[l.name] = (verts, offsets)

    walls = None
    if wall_mesh and scene.walls:
        wv, wt = concat_meshes(scene.walls)
        walls = (swap_to_world(wv), wt)
    return height, phase, layers, walls


def export(scene, path, *, chunk_count=1, phase=True, wall_mesh=True, derived=False,
           example=""):
    """Serialize a `Scene` to `path` and return the `Manifest` written."""
    manifest = manifest_of(scene, phase=phase, chunk_count=chunk_count,
                           wall_mesh=wall_mesh, derived=derived, example=example)
    height, ph, layers, walls = arrays_of(scene, manifest, wall_mesh=wall_mesh)
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
                    help="describe rather than dump: the walls become a "
                         "perimeter, and every layer that can be stated as "
                         "levels of a field becomes that statement")
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
                      phase=not args.no_phase, wall_mesh=not args.derived,
                      derived=args.derived, example=args.example)

    if not args.quiet:
        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"wrote {out} ({total / 1e6:.1f} MB)")
        for spec in manifest.layers:
            files = spec.files
            if files is None:
                what = f"{len(spec.source.levels)} levels of {spec.source.field}"
                if spec.source.tiled:
                    what += f" x{len(manifest.occluder.tiles)}"
                detail = f"{what:>26} (derived)"
            else:
                detail = f"{len(np.load(out / files[1])) - 1:>7} paths          "
            print(f"  {spec.name:<12} {detail}  lw={spec.width}"
                  f"{'' if spec.clipped else '  (unclipped)'}")


if __name__ == "__main__":
    main()
