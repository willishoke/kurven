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
    contour/   contourpy's iso-lines on three awkward grids -- nested loops,
               nothing but saddles, and a step function -- so the native
               marching squares can be held to them.
    hatch/     the four hatching sources derived from one grid, one perimeter
               and two kinds of cap, by `kurven.hatch` -- which is their
               definition. The consumer derives the same strokes from the same
               description, and this is where the two are compared. Heights
               here come from the *grid*, not from f, because that is all the
               consumer has; the plate's own hatching uses f, and the size of
               that difference is `tests/compare_bake.py --derived`.

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

from kurven.contours import contour_levels  # noqa: E402
from kurven.bundle import (  # noqa: E402
    AXES,
    LayerCapHatch,
    LayerCapOutline,
    LayerWallHatch,
    LayerWallOutline,
    Affine2,
    CameraPreset,
    Domain,
    Edge,
    FullRegion,
    GridRef,
    Interval,
    InsideRegion,
    KeepAll,
    KeepBand,
    KeepBelowCap,
    KeepEvery,
    KeepRegion,
    LayerContour,
    LayerFile,
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


#: The hatching fixture's landscape: a grid small enough to read and awkward
#: enough to matter. One pole sits just inside the back edge, so a wall crest
#: runs into the cap and the two truncations disagree about where it stops; the
#: other is interior, so there is a plateau with a rim all the way round. Both
#: are off the sample lattice, because a pole *on* a sample is an infinity in
#: the grid and this fixture is about hatching, not about sanitizing.
HATCH_DOMAIN = Domain(Interval(-2.0, 2.5), Interval(-1.0, 1.5))
HATCH_SHAPE = (37, 23)          # (nx, ny): deliberately not square


def _hatch_surface():
    from kurven.expr import compile_expression

    nx, ny = HATCH_SHAPE
    real = np.linspace(HATCH_DOMAIN.real.lo, HATCH_DOMAIN.real.hi, nx)
    imag = np.linspace(HATCH_DOMAIN.imag.lo, HATCH_DOMAIN.imag.hi, ny)
    f = compile_expression("1/((z - 1.05 - 1.44i)(z + 0.7 - 0.3i))")
    values = f(real[:, None] + 1j * imag[None, :])
    return real, imag, np.abs(values).astype(np.float32).T      # (ny, nx)


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
    """Four tiny bundles that between them use every arm of `Caps`, `Walls`,
    `Region` and `LayerSource`, a phase grid and no phase grid, tiles and no
    tiles, a clipped and an unclipped layer, a preset and no presets."""
    height, phase = _tiny_grid()
    ny, nx = height.shape
    domain = Domain(Interval(-1.0, 2.0), Interval(0.0, 1.0))

    # (1) uniform cap, explicit wall mesh, two tiles, both layer kinds.
    layers = (
        LayerSpec("mag", "magnitude",
                  LayerFile("layers/mag.npy", "layers/mag.idx.npy"), 0.4, "level"),
        LayerSpec("hatch", "scaffold",
                  LayerFile("layers/hatch.npy", "layers/hatch.idx.npy"),
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
                      RealBand(-1.5, 4.0)), 3.0),
        Occluder(1, (Affine2.identity(),), WallPerimeter(perim, 0.0),
                 InsideRegion(perim), 0.0),
        # A described layer alongside a dumped one, so the contract fixture
        # covers both arms of LayerSource and the whole Keep vocabulary.
        (LayerSpec("ang", "phase",
                   LayerFile("layers/ang.npy", "layers/ang.idx.npy"),
                   0.15, "magnitude"),
         LayerSpec("mag_derived", "magnitude",
                   LayerContour("magnitude", (0.5, 1.0, 1.5),
                                KeepEvery((KeepRegion(), KeepBelowCap(),
                                           KeepBand("imag", -1.0, 1.0)))),
                   0.3, "level"),
         LayerSpec("all_levels", "magnitude",
                   LayerContour("phase", (0.25,), KeepAll(), tiled=True),
                   0.2, "surface")),
        tuple(CameraPreset(n, p, 0.01, 1024) for n, p in PRESETS.items()),
        _provenance("tiny_bands"))
    write_bundle(out / "bands_perimeter.kurven", manifest=m, height=height,
                 layers={"ang": _tiny_layer(1, 4, 3)})

    # (3) a landscape's own shape: every layer described, and the four
    #     hatching kinds alongside a contour family. This is the bundle
    #     `kurven.landscape` writes, shrunk to nothing, and it is what keeps
    #     the hatching sources in the round-trip test.
    m = Manifest(
        SCHEMA, AXES, domain,
        GridRef("height.npy", (ny, nx), "<f4"),
        GridRef("phase.npy", (ny, nx), "<f4"),
        UniformCap(1.75),
        Occluder(1, (Affine2.identity(),), WallPerimeter(perim, 0.0),
                 FullRegion(), 0.0),
        (LayerSpec("mag_major", "magnitude",
                   LayerContour("magnitude", (0.5, 1.0, 1.5), KeepBelowCap()),
                   0.4, "level"),
         LayerSpec("cap_outline", "scaffold", LayerCapOutline(), 0.4, "level"),
         LayerSpec("cap_hatch", "scaffold", LayerCapHatch("real", 0.25), 0.2, "level"),
         LayerSpec("cap_hatch_tiled", "scaffold",
                   LayerCapHatch("imag", 0.4, tiled=True), 0.2, "level"),
         LayerSpec("wall_outline", "scaffold",
                   LayerWallOutline((0, 1, 2, 3), 0.1, 0.0), 0.3, "surface"),
         LayerSpec("wall_hatch", "scaffold",
                   LayerWallHatch((0, 2), 0.3, 0.1, True, 0.01, -0.02),
                   0.25, "surface")),
        (CameraPreset("plate", PRESETS["recip"], 0.02, 4000),),
        _provenance("1/gamma(z)", expression="1/gamma(z)", resolution=4),
    )
    write_bundle(out / "landscape.kurven", manifest=m, height=height, phase=phase,
                 layers={})

    # (4) the empty case: no caps, no walls, no layers, no presets.
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
# contour
# --------------------------------------------------------------------------


def contour_fixture(out):
    """contourpy's answer on grids chosen to exercise the awkward cases.

    Three fields, contoured at levels that are deliberately not round numbers
    relative to the data:

      ripple   smooth, many nested closed loops, contours that leave the grid
      saddles  a product of sines: nothing but saddle points, which is the one
               case marching squares has to make a choice about
      cliff    a step function, so contours run along cell boundaries and every
               interpolation lands at an endpoint

    Stored as one flat array per level plus path offsets, in domain coordinates.
    """
    n = 61
    real = np.linspace(-3.0, 3.0, n)
    imag = np.linspace(-2.0, 2.0, n)
    R, I = np.meshgrid(real, imag, indexing="ij")

    fields = {
        "ripple": np.sin(1.7 * R) * np.cos(2.3 * I) + 0.35 * R,
        "saddles": np.sin(3.0 * R) * np.sin(3.0 * I),
        "cliff": np.where(R + 0.5 * I > 0.3, 1.0, -1.0) + 0.001 * I,
    }
    levels = {
        "ripple": [-0.9, -0.31, 0.0, 0.17, 0.62, 1.05],
        "saddles": [-0.5, -0.02, 0.0, 0.02, 0.5],
        "cliff": [0.0, 0.5],
    }

    index = {}
    for name, field in fields.items():
        # Contour the float32 array, not the float64 one it came from: the Swift
        # side reads float32 out of the bundle, and two implementations
        # interpolating between different numbers disagree for a reason that has
        # nothing to do with either of them.
        field = field.astype(np.float32)
        save(out / f"{name}.npy", field.T)                      # (ny, nx), world order
        for lvl in levels[name]:
            paths = contour_levels(field, [lvl], (-3.0, 3.0), (-2.0, 2.0), chunk_count=1)
            segs = [np.asarray(s) for _, ss in paths for s in ss if len(s) >= 2]
            tag = f"{name}_{lvl}".replace("-", "m").replace(".", "p")
            if segs:
                save(out / f"{tag}.npy", np.concatenate(segs))
                save(out / f"{tag}.idx.npy",
                     np.concatenate([[0], np.cumsum([len(s) for s in segs])]).astype(np.int64))
            else:
                save(out / f"{tag}.npy", np.zeros((0, 2)))
                save(out / f"{tag}.idx.npy", np.zeros(1, dtype=np.int64))
            index.setdefault(name, []).append({"level": lvl, "tag": tag,
                                               "paths": len(segs)})

    (out / "index.json").write_text(json.dumps({
        "grid": {"nx": n, "ny": n, "real": [-3.0, 3.0], "imag": [-2.0, 2.0]},
        "fields": index,
        # contourpy returns (imag, real); the arrays saved here are in that
        # order, and the Swift side works in world (real, imag).
        "columns": ["imag", "real"],
    }, sort_keys=True, indent=1) + "\n")


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
# hatch
# --------------------------------------------------------------------------


def hatch_fixtures(out):
    """Every hatching source, derived from one grid by `kurven.hatch`.

    The grid is written as float32 and every height is read back out of it with
    `grid_sampler` -- `Grid2D.sample` transcribed -- so this fixture is a
    question the consumer can answer exactly rather than within a tolerance.
    Each case names its caps and its source; the expected strokes are CSR, the
    same shape a layer is.
    """
    from kurven import hatch

    real, imag, height = _hatch_surface()
    save(out / "height.npy", height)

    perimeter = Perimeter((
        Edge((HATCH_DOMAIN.real.lo, HATCH_DOMAIN.imag.lo),
             (HATCH_DOMAIN.real.hi, HATCH_DOMAIN.imag.lo), 19),
        Edge((HATCH_DOMAIN.real.hi, HATCH_DOMAIN.imag.lo),
             (HATCH_DOMAIN.real.hi, HATCH_DOMAIN.imag.hi), 11),
        Edge((HATCH_DOMAIN.real.hi, HATCH_DOMAIN.imag.hi),
             (HATCH_DOMAIN.real.lo, HATCH_DOMAIN.imag.hi), 19),
        Edge((HATCH_DOMAIN.real.lo, HATCH_DOMAIN.imag.hi),
             (HATCH_DOMAIN.real.lo, HATCH_DOMAIN.imag.lo), 11),
    ))
    caps = {
        "uniform": UniformCap(2.0),
        # A staircase whose boundary falls between the two poles, so a cap
        # stroke has to stop at the band edge and start again at another height.
        "bands": RealBandCaps((RealBand(-1.0, 1.2), RealBand(0.4, 2.5)), 1.6),
    }
    sources = {
        "wall_hatch": LayerWallHatch((0, 1, 2, 3), 0.35, 0.5, True, 0.0, 0.0),
        "wall_hatch_untrimmed": LayerWallHatch((0, 2), 0.6, 0.25, False, 0.1, -0.05),
        "wall_outline": LayerWallOutline((0, 1, 2, 3), 0.4, 0.0),
        "cap_hatch_real": LayerCapHatch("real", 0.25),
        "cap_hatch_imag": LayerCapHatch("imag", 0.3),
        "cap_outline": LayerCapOutline(),
    }

    sampler = hatch.grid_sampler(height, HATCH_DOMAIN)
    index = {"domain": HATCH_DOMAIN.to_dict(),
             "shape": {"ny": int(height.shape[0]), "nx": int(height.shape[1])},
             "perimeter": perimeter.to_dict(),
             "caps": {k: v.to_dict() for k, v in caps.items()},
             "cases": []}
    for cap_name, cap in caps.items():
        def capped(x, y, cap=cap):
            return np.minimum(sampler(x, y), cap.at(x))

        for source_name, source in sources.items():
            paths = hatch.derive(source, perimeter=perimeter, region=None,
                                 tiles=(Affine2.identity(),), height=capped,
                                 magnitude=sampler, height_grid=height,
                                 domain=HATCH_DOMAIN, caps=cap, chunk_count=1)
            stem = f"{cap_name}_{source_name}"
            verts = np.vstack(paths) if paths else np.zeros((0, 3))
            offsets = np.concatenate([[0], np.cumsum([len(p) for p in paths])])
            save(out / f"{stem}.npy", verts)
            save(out / f"{stem}.idx.npy", offsets.astype(np.int64))
            index["cases"].append({
                "name": stem, "caps": cap_name, "source": source.to_dict(),
                "vertices": f"{stem}.npy", "offsets": f"{stem}.idx.npy",
                "paths": len(paths)})

    # The two derived numbers the consumer computes for itself, because a cap
    # it moves has to bring the contour levels with it. Pinned here so the two
    # implementations of one rule cannot drift.
    from kurven.landscape import magnitude_levels, nice

    index["style"] = {
        "nice": [{"x": x, "up": nice(x), "down": nice(x, down=True)}
                 for x in (0.037, 0.12, 0.5, 1.0, 1.4, 2.3, 4.9, 6.0, 7.5, 23.0, 180.0)],
        "levels": [{"ceiling": c,
                    "major": list(magnitude_levels(c)[0]),
                    "minor": list(magnitude_levels(c)[1])}
                   for c in (1.0, 2.0, 4.0, 5.0, 6.0, 10.0, 12.5, 37.0)],
    }
    (out / "index.json").write_text(json.dumps(index, sort_keys=True, indent=1) + "\n")
    return sum(c["paths"] for c in index["cases"])



# --------------------------------------------------------------------------
# the expression language and its functions
# --------------------------------------------------------------------------


def expr_fixtures(out):
    """The expression language, its forty-three functions, and the landscapes
    built over them, as the Python side answers them.

    Three things are pinned, and they fail in different ways:

      values     every function of the table on a grid that runs along the real
                 axis at exactly im = 0 (the branch cuts), plus each special
                 function on the window a landscape of it uses. The Swift side
                 has its own implementation of every one of these -- scipy's
                 algorithm where scipy has one, `kurven.expr`'s where it does
                 not -- and this is where the two are compared, to a tolerance
                 stated per function.
      parsing    the canonical spelling of the ambiguous cases, and the
                 character each malformed expression is refused at.
      landscape  two specs, built by `kurven.landscape` with geometry left
                 described, written as bundles. The native builder must produce
                 the same manifest (but for the git sha) and the same grids.
    """
    import dataclasses

    from kurven.expr import FUNCTIONS, LANGUAGE, ExpressionError, canonical, compile_expression, parse
    from kurven.landscape import (CATALOG, DEFAULT_RESOLUTION, PLATE_BUFFER, Spec,
                                  build_scene)
    from kurven.export import export

    def evaluate(text, real, imag):
        r = np.linspace(real[0], real[1], real[2])
        i = np.linspace(imag[0], imag[1], imag[2])
        return compile_expression(text)(r[:, None] + 1j * i[None, :])

    grid = {"real": [-3.5, 3.5, 15], "imag": [-2.5, 2.5, 13]}
    cases = []

    def case(name, text, tol, real=None, imag=None):
        real = real or grid["real"]
        imag = imag or grid["imag"]
        values = evaluate(text, real, imag)
        np.save(out / f"{name}.npy", np.ascontiguousarray(values.astype(np.complex128)))
        cases.append({"name": name, "expression": text, "real": real, "imag": imag,
                      "file": f"{name}.npy", "tol": tol})

    # Every name in the table, at the spelling `tests/check_expr.py` uses.
    loose = {"besselj", "bessely", "besseli", "besselk", "airyai", "airybi", "zeta"}
    for name, f in sorted(FUNCTIONS.items()):
        text = f"{name}(z)" if f.arity == 1 else f"{name}(0.5, z)"
        if name in ("sn", "cn", "dn"):
            text = f"{name}(z, 0.64)"
        case(f"table_{name}", text, 1e-9 if name in loose else 1e-12)

    # The special functions on the windows their landscapes use, and a little
    # beyond: the regimes each implementation switches between are in here.
    wide = [
        ("gamma_wide", "gamma(z)", 1e-12, [-12, 12, 49], [-6, 6, 25]),
        ("rgamma_wide", "rgamma(z)", 1e-12, [-12, 12, 49], [-6, 6, 25]),
        ("loggamma_wide", "loggamma(z)", 1e-12, [-12, 12, 49], [-6, 6, 25]),
        ("digamma_wide", "digamma(z)", 1e-12, [-12, 12, 49], [-6, 6, 25]),
        ("zeta_wide", "zeta(z)", 1e-10, [-5, 8, 27], [-30, 30, 31]),
        ("erf_wide", "erf(z)", 1e-12, [-8, 8, 33], [-8, 8, 33]),
        ("erfc_wide", "erfc(z)", 1e-12, [-8, 8, 33], [-8, 8, 33]),
        ("erfi_wide", "erfi(z)", 1e-12, [-8, 8, 33], [-8, 8, 33]),
        ("wofz_wide", "wofz(z)", 1e-12, [-8, 8, 33], [-8, 8, 33]),
        ("expi_wide", "expi(z)", 1e-10, [-25, 25, 51], [-25, 25, 51]),
        ("lambertw_wide", "lambertw(z)", 1e-10, [-8, 8, 33], [-8, 8, 33]),
        ("besselj0_wide", "besselj(0, z)", 1e-9, [-25, 25, 51], [-8, 8, 17]),
        ("besselj10_wide", "besselj(10, z)", 1e-9, [-30, 30, 31], [-30, 30, 31]),
        ("bessely2p7_wide", "bessely(2.7, z)", 1e-9, [-30, 30, 31], [-30, 30, 31]),
        ("besseli1p5_wide", "besseli(1.5, z)", 1e-9, [-25, 25, 51], [-8, 8, 17]),
        ("besselk5_wide", "besselk(5, z)", 1e-9, [-30, 30, 31], [-30, 30, 31]),
        ("besselkm2p3_wide", "besselk(-2.3, z)", 1e-9, [-25, 25, 26], [-25, 25, 26]),
        ("airyai_wide", "airyai(z)", 1e-9, [-12, 12, 49], [-12, 12, 49]),
        ("airybi_wide", "airybi(z)", 1e-9, [-12, 12, 49], [-12, 12, 49]),
        ("cn_wide", "cn(z, 0.9)", 1e-12, [-6, 6, 25], [-4, 4, 17]),
        ("sn_wide", "sn(z, 0.2)", 1e-12, [-6, 6, 25], [-4, 4, 17]),
        ("tan_wide", "tan(z)", 1e-12, [-12, 12, 49], [-4, 4, 17]),
        ("sinh_wide", "sinh(z)", 1e-12, [-12, 12, 49], [-12, 12, 49]),
    ]
    for name, text, tol, real, imag in wide:
        case(name, text, tol, real, imag)

    # The readings two reasonable people disagree about, as Python reads them.
    precedence = ["-z^2", "2^-z", "2z^2", "z^2^3", "(z^2)^3", "1/2z", "z(z+1)",
                  "(z-1)(z+1)", "2 pi i", "2sin(z)", "1e-3z", "2e", "z - -z", "--z",
                  "z!", "(z+1)!", "1/Γ(z)", "ζ(z) + ψ(z)", "ln(z) + arcsin(z)",
                  "z!^2", "exp(1/z)", "(z-1)(z+1)/(z^2+1)", "-z^2+2z-1/(z^2+1)",
                  "2**z", "abs(z)^2 + re(z)*im(z) - arg(conj(z))", "1e16z", "0.000125z",
                  "sqrt(z) + log(z) + acos(z) + atanh(z) + acosh(z) + asinh(z) + atan(z)"]
    for k, text in enumerate(precedence):
        case(f"reading_{k}", text, 1e-12)

    canonical_cases = ["1 / Γ(z)", "-z^2", "2z^2", "z(z+1)", "2**z", "(z-1)(z+1)", "ln(z)",
                       "1/Γ(z)", "-z^2+2z-1/(z^2+1)", "cn(z, 0.64)", "z!^2", "2^-z",
                       "exp(1/z)", "(z-1)(z+1)/(z^2+1)", "2 pi i", "1e-3z", "0.5e", "1e16z",
                       "besselj(0.5, z)", "z^(1/3)", "-(-z)", "(-z)^2", "-z!", "z^-1",
                       "1e-05 z", "123456789012345.6 z", "2.50z"]
    canonical_spellings = {text: canonical(text) for text in canonical_cases}

    errors = []
    for text in ["", "sin(z", "(z+1", "z +", "foo(z)", "gamma(z, 2)", "z..2", "gamma",
                 "z 2 @", ")", "gama(z)", "1/", "z^", "cn(z)", "2 PI",
                 "__import__(\"os\")", "z; print(1)"]:
        try:
            parse(text)
            errors.append({"text": text, "parses": True})
        except ExpressionError as e:
            errors.append({"text": text, "position": e.position, "length": e.length,
                           "message": e.message})

    # Two landscapes, built the way the service builds them: every layer a
    # description, no walls dumped. One leaves the truncation to the rule, the
    # other says where to cut and how densely to hatch.
    landscapes = []
    for stem, spec in [("landscape_gamma", Spec.of("gamma", resolution=90)),
                       ("landscape_cubic", Spec.of("cubic", resolution=64, spacing=0.4)),
                       ("landscape_typed",
                        Spec(expression="sin(z)/z", resolution=48,
                             domain=Spec.of("sin").domain))]:
        scene = build_scene(spec, verbose=False, geometry=False, chunk_count=1)
        scene = dataclasses.replace(scene, walls=())
        manifest = export(scene, out / f"{stem}.kurven", chunk_count=1, phase=True,
                          wall_mesh=False, derived=True, example="function")
        landscapes.append({"bundle": f"{stem}.kurven", "spec": spec.to_dict(),
                           "layers": len(manifest.layers)})

    (out / "index.json").write_text(json.dumps({
        "grid": grid,
        "columns": "[real][imag]",
        "cases": cases,
        "canonical": canonical_spellings,
        "errors": errors,
        "language": LANGUAGE,
        "catalog": [p.to_dict() for p in CATALOG],
        "defaults": {"resolution": DEFAULT_RESOLUTION, "buffer": PLATE_BUFFER},
        "landscapes": landscapes,
    }, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(cases)


# --------------------------------------------------------------------------


def main():
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    for sub in ("contract", "npy", "camera", "clip", "contour", "hatch", "expr"):
        (FIXTURES / sub).mkdir(parents=True)

    contract_fixtures(FIXTURES / "contract")
    contour_fixture(FIXTURES / "contour")
    npy_fixtures(FIXTURES / "npy")
    camera_fixtures(FIXTURES / "camera")
    n = clip_fixture(FIXTURES / "clip")
    strokes = hatch_fixtures(FIXTURES / "hatch")
    expressions = expr_fixtures(FIXTURES / "expr")

    total = sum(f.stat().st_size for f in FIXTURES.rglob("*") if f.is_file())
    print(f"wrote {FIXTURES.relative_to(ROOT)} "
          f"({total / 1e3:.0f} kB, clip has {n} visible segments, "
          f"hatch has {strokes} strokes, expr has {expressions} cases)")


if __name__ == "__main__":
    main()
