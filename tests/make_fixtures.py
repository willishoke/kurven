"""Generate `tests/fixtures/` — the oracle the Swift frontend is tested against.

Correctness of the frontend is anchored on this pipeline, not on a second
opinion. Every fixture here is a Python-computed answer to a question the Swift
side must answer identically, cheapest first:

    contract/  tiny bundles exercising every arm of every sum type. Decode,
               re-encode the manifest, compare canonical JSON. This is the only
               test that the two schema definitions agree, and it is why the
               Swift mirror needs no codegen.
    npy/       one file per supported dtype, plus the files a reader must
               *reject* (Fortran order, an unsupported dtype).
    camera/    world points and their projections under each plate preset. This
               is the test that makes "the bake reproduces the plate" credible
               before any GPU code exists.
    clip/      a depth buffer, view-space polylines, and the visible segments
               `clip_hidden_lines` produces from them. Pure, no GPU in the loop.

Determinism: contouring runs single-chunk, the point sets come from a seeded
generator, and the depth buffer is stored as float32 with the expected clip
computed from the *rounded* buffer, so the two sides compare the same numbers
rather than two roundings of them.

    python tests/make_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.special as ss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kurven.bundle import (  # noqa: E402
    AXES,
    Affine2,
    CameraPreset,
    Domain,
    Edge,
    FullRegion,
    GridRef,
    Interval,
    InsideRegion,
    LayerSpec,
    Manifest,
    NoCaps,
    NoWalls,
    Occluder,
    Perimeter,
    PlateProjection,
    Provenance,
    RealBand,
    RealBandCaps,
    SCHEMA,
    UniformCap,
    WallMesh,
    WallPerimeter,
    write_bundle,
)
from kurven.occluder import build_occluder  # noqa: E402
from kurven.outline import clip_hidden_lines  # noqa: E402
from kurven.perimeter import Perimeter as SurfacePerimeter  # noqa: E402
from kurven.projection import Projection  # noqa: E402
from kurven.surface import Surface  # noqa: E402
from kurven.zbuffer import ZBuffer, rasterize_triangles  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def save(path, array):
    """`np.save`, C-contiguous.

    scipy's `Rotation.apply` returns Fortran-order arrays, and `np.save` records
    that faithfully. The Swift reader refuses Fortran order rather than silently
    reinterpreting it as C order and returning a transposed landscape -- which is
    correct of it, and means the fixtures must say what they mean.
    """
    np.save(path, np.ascontiguousarray(array))

#: The three plate cameras, as the examples set them. Kept here rather than
#: imported so a change to an example is a *visible* fixture change.
PRESETS = {
    "recip": PlateProjection(0.5, -55.0, -90.0, True, None),
    "elliptic": PlateProjection(0.51, -63.0, -90.0, True, None),
    "zeta": PlateProjection(-0.18, -79.5, -90.0, True, 0.75),
}


def _provenance(name, **params):
    return Provenance(name, params, 1, "fixture")


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


def _tiny_grid(ny=3, nx=4):
    y = np.linspace(0.0, 1.0, ny)[:, None]
    x = np.linspace(-1.0, 2.0, nx)[None, :]
    return (x * x + y).astype(np.float32), (x - y).astype(np.float32)


def _tiny_layer(n_paths, n_pts, seed):
    rng = np.random.default_rng(seed)
    verts = rng.normal(size=(n_paths * n_pts, 3))
    offsets = np.arange(n_paths + 1, dtype=np.int64) * n_pts
    return verts, offsets


def contract_fixtures(out):
    """Three tiny bundles that between them use every arm of `Caps` and
    `Walls`, a phase grid and no phase grid, tiles and no tiles, a clipped and
    an unclipped layer, a preset and no presets."""
    height, phase = _tiny_grid()
    ny, nx = height.shape
    domain = Domain(Interval(-1.0, 2.0), Interval(0.0, 1.0))

    # (1) uniform cap, explicit wall mesh, two tiles, both layer kinds.
    layers = (
        LayerSpec("mag", "magnitude", "layers/mag.npy", "layers/mag.idx.npy",
                  0.4, "level"),
        LayerSpec("hatch", "scaffold", "layers/hatch.npy", "layers/hatch.idx.npy",
                  0.25, "surface", "#333333", False),
    )
    m = Manifest(
        SCHEMA, AXES, domain,
        GridRef("height.npy", (ny, nx), "<f4"),
        GridRef("phase.npy", (ny, nx), "<f4"),
        UniformCap(2.5),
        Occluder(2,
                 (Affine2.identity(), Affine2.scale_offset(-1.0, 1.0, 3.0, 0.0)),
                 WallMesh("occluder/walls.npy", "occluder/walls.tri.npy"),
                 FullRegion(), 0.0),
        layers,
        (CameraPreset("recip", PRESETS["recip"], 0.02, 512),),
        _provenance("tiny", res=4))
    wv = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]])
    wt = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    write_bundle(out / "uniform_mesh.kurven", manifest=m, height=height,
                 phase=phase,
                 layers={"mag": _tiny_layer(2, 5, 1),
                         "hatch": _tiny_layer(3, 2, 2)},
                 walls=(wv, wt))

    # (2) real-band caps, walls described as a perimeter, no phase, no tiles
    #     beyond the identity, several presets.
    perim = Perimeter((
        Edge((-1.0, 0.0), (2.0, 0.0), 8),
        Edge((2.0, 0.0), (2.0, 1.0), 4),
        Edge((2.0, 1.0), (-1.0, 1.0), 8),
        Edge((-1.0, 1.0), (-1.0, 0.0), 4),
    ))
    m = Manifest(
        SCHEMA, AXES, domain,
        GridRef("height.npy", (ny, nx), "<f4"), None,
        RealBandCaps((RealBand(-3.5, 8.0), RealBand(-2.5, 6.0),
                      RealBand(-1.5, 4.0))),
        Occluder(1, (Affine2.identity(),), WallPerimeter(perim, 0.0),
                 InsideRegion(perim), 0.0),
        (LayerSpec("ang", "phase", "layers/ang.npy", "layers/ang.idx.npy",
                   0.15, "magnitude"),),
        tuple(CameraPreset(n, p, 0.01, 1024) for n, p in PRESETS.items()),
        _provenance("tiny_bands"))
    write_bundle(out / "bands_perimeter.kurven", manifest=m, height=height,
                 layers={"ang": _tiny_layer(1, 4, 3)})

    # (3) the empty case: no caps, no walls, no layers, no presets.
    m = Manifest(
        SCHEMA, AXES, domain,
        GridRef("height.npy", (ny, nx), "<f4"), None,
        NoCaps(),
        Occluder(1, (Affine2.identity(),), NoWalls(), FullRegion(), 0.0),
        (), (), _provenance("tiny_empty"))
    write_bundle(out / "empty.kurven", manifest=m, height=height, layers={})


# --------------------------------------------------------------------------
# npy
# --------------------------------------------------------------------------


def npy_fixtures(out):
    """One file per dtype the reader must accept, and two it must reject with a
    typed error rather than a wrong answer."""
    rng = np.random.default_rng(7)
    save(out / "f4.npy", rng.normal(size=(3, 5)).astype(np.float32))
    save(out / "f8.npy", rng.normal(size=(4, 2)).astype(np.float64))
    save(out / "i8.npy", rng.integers(-2**40, 2**40, size=(6,)).astype(np.int64))
    save(out / "c16.npy",
            (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))).astype(np.complex128))
    save(out / "f8_1d.npy", np.arange(7, dtype=np.float64))
    # Rejected: Fortran order, and an unsupported dtype.
    np.save(out / "reject_fortran.npy", np.asfortranarray(rng.normal(size=(3, 3))))
    save(out / "reject_dtype.npy", rng.integers(0, 200, size=(4,)).astype(np.uint8))


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


def _sample_points(seed=11, n=256):
    """A point cloud that exercises the whole transform: random interior points
    plus the corners of a box, so a sign error anywhere is visible."""
    rng = np.random.default_rng(seed)
    lo = np.array([-6.0, -30.0, 0.0])
    hi = np.array([8.0, 30.0, 6.0])
    inner = rng.uniform(lo, hi, size=(n, 3))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return np.vstack([corners, inner])


def camera_fixtures(out):
    """World points (x = real, y = imag, z) and their projections.

    `Projection.apply` consumes the library's `(imag, real, z)`, so the fixture
    records the *world* points and the projection of their exchanged form. A
    consumer working in world order must fold the exchange into its camera
    matrix; that is exactly what this fixture pins down.
    """
    world = _sample_points()
    legacy = world[:, [1, 0, 2]]
    for name, plate in PRESETS.items():
        proj = Projection(shear=plate.shear, x_angle=plate.x_angle,
                          z_angle=plate.z_angle, flip_x=plate.flip_x,
                          y_scale=plate.y_scale)
        save(out / f"{name}.points.npy", world)
        save(out / f"{name}.projected.npy", proj.apply(legacy))
        (out / f"{name}.json").write_text(json.dumps(
            {"name": name, "plate": plate.to_dict()},
            sort_keys=True, separators=(",", ":")) + "\n")


# --------------------------------------------------------------------------
# clip
# --------------------------------------------------------------------------


def clip_fixture(out, *, res=200, occluder_res=100, buffer=320, margin=0.02):
    """A real hidden-line problem, small enough to check in.

    1/Γ on a coarse grid, its occluder rasterized on the CPU into a `buffer²`
    Z-buffer, and the visible segments `clip_hidden_lines` extracts from a set
    of view-space polylines. The buffer is written as float32 and the expected
    answer recomputed from that rounded buffer, so a consumer reading the file
    is comparing against the numbers it actually has.
    """
    real = np.linspace(-5.5, 4.0, res)
    imag = np.linspace(0.0, 2.5, res)
    surface = Surface.from_function(ss.rgamma, real, imag, z_limit=5.0)

    perim = SurfacePerimeter.rectangle((0.0, 2.5), (-5.5, 4.0))
    occ_v, occ_t = build_occluder(surface, max(1, res // occluder_res),
                                  walls=perim.wall_curtains(surface, 120))

    project = Projection(shear=0.5, x_angle=-55.0, z_angle=-90.0, flip_x=True)
    occ_rot = project(occ_v)

    # Polylines to clip: a lattice of straight probes across the domain, lifted
    # onto the surface. Straight lines make it obvious which runs should split.
    rng = np.random.default_rng(23)
    paths = []
    for _ in range(60):
        a = np.array([rng.uniform(0.0, 2.5), rng.uniform(-5.5, 4.0)])
        b = np.array([rng.uniform(0.0, 2.5), rng.uniform(-5.5, 4.0)])
        t = np.linspace(0, 1, 40)[:, None]
        xy = a[None, :] * (1 - t) + b[None, :] * t
        z = surface.height_at(xy[:, 1], xy[:, 0])
        paths.append(np.column_stack([xy, z]))
    xyz = np.concatenate(paths)
    indices = np.concatenate([np.full(len(p), i, dtype=np.int64)
                              for i, p in enumerate(paths)])
    rot = project(xyz)

    xs = np.concatenate([occ_rot[:, 0], rot[:, 0]])
    ys = np.concatenate([occ_rot[:, 1], rot[:, 1]])
    zb = ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), (buffer, buffer))
    rasterize_triangles(zb, occ_t, occ_rot[:, 0], occ_rot[:, 1], occ_rot[:, 2])

    # Round the buffer to float32 *before* clipping, so file and answer agree.
    zb.buffer = zb.buffer.astype(np.float32).astype(np.float64)
    segments = clip_hidden_lines(zb, rot, indices, margin=margin)

    save(out / "depth.npy", zb.buffer.astype(np.float32))
    save(out / "view.npy", rot)
    save(out / "view.idx.npy",
            np.concatenate([[0], np.cumsum([len(p) for p in paths])]).astype(np.int64))
    if segments:
        save(out / "expected.npy", np.concatenate(segments))
        save(out / "expected.idx.npy",
                np.concatenate([[0], np.cumsum([len(s) for s in segments])]
                               ).astype(np.int64))
    else:
        save(out / "expected.npy", np.zeros((0, 2)))
        save(out / "expected.idx.npy", np.zeros(1, dtype=np.int64))

    (out / "meta.json").write_text(json.dumps({
        # ZBuffer's mapping, verbatim. `coordToIndex` is
        #   floor(0.01 + (shape - 1) * (coord - lower) / imageSize)
        # and the 0.01 nudge is part of the contract: it keeps a coordinate
        # exactly on `lower` from flooring to -1 under float jitter.
        "axis0": [float(zb.lower[0]), float(zb.lower[0] + zb.image_size[0])],
        "axis1": [float(zb.lower[1]), float(zb.lower[1] + zb.image_size[1])],
        "shape": [int(buffer), int(buffer)],
        "nudge": 0.01,
        "margin": float(margin),
        "paths": len(paths),
        "segments": len(segments),
    }, sort_keys=True, indent=1) + "\n")
    return len(segments)


# --------------------------------------------------------------------------


def main():
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    for sub in ("contract", "npy", "camera", "clip"):
        (FIXTURES / sub).mkdir(parents=True)

    contract_fixtures(FIXTURES / "contract")
    npy_fixtures(FIXTURES / "npy")
    camera_fixtures(FIXTURES / "camera")
    n = clip_fixture(FIXTURES / "clip")

    total = sum(f.stat().st_size for f in FIXTURES.rglob("*") if f.is_file())
    print(f"wrote {FIXTURES.relative_to(ROOT)} "
          f"({total / 1e3:.0f} kB, clip has {n} visible segments)")


if __name__ == "__main__":
    main()
