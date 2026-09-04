"""The Python lane of the bundle contract tests.

`tests/make_fixtures.py` writes the fixtures; this checks that the Python side
of the contract holds on them, so a schema mistake is caught here rather than
in Swift, where the failure would look like a decoder bug. The Swift lane runs
the same assertions on the same files (`KurvenCoreTests`), which is what makes
them a contract rather than two independent opinions.

    python tests/check_bundle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kurven.bundle import (  # noqa: E402
    Affine2,
    BundleError,
    Manifest,
    WallMesh,
    WallPerimeter,
    csr_from_indices,
    indices_from_csr,
    read_bundle,
    swap_to_world,
)
from kurven.perimeter import Edge, Perimeter  # noqa: E402
from kurven.projection import Projection  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

_failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def check_manifest_roundtrip():
    """A manifest decoded and re-encoded must be byte-identical. This is what
    the Swift mirror is held to, so it must hold here first."""
    for bundle in sorted((FIXTURES / "contract").iterdir()):
        text = (bundle / "manifest.json").read_text()
        again = Manifest.from_json(text).to_json()
        check(f"contract/{bundle.name} canonical round trip", again == text)


def check_bundle_reads():
    for bundle in sorted((FIXTURES / "contract").iterdir()):
        b = read_bundle(bundle)
        m = b.manifest
        ok = b.height.shape == tuple(m.height.shape)
        ok &= (b.phase is None) == (m.phase is None)
        ok &= len(b.layers) == len(m.layers)
        for layer in b.layers:
            ok &= layer.offsets[-1] == len(layer.vertices)
            ok &= all(len(p) >= 2 for p in layer.paths)
        if isinstance(m.occluder.walls, WallMesh):
            ok &= b.wall_triangles.max() < len(b.wall_vertices)
        if isinstance(m.occluder.walls, WallPerimeter):
            ok &= len(m.occluder.walls.perimeter.edges) > 0
        check(f"contract/{bundle.name} decodes consistently", bool(ok))


def check_schema_rejection():
    """A reader must refuse what it does not understand, with a typed error."""
    good = json.loads((FIXTURES / "contract" / "empty.kurven" / "manifest.json").read_text())
    for name, mutate in [
        ("wrong schema", lambda d: d.update(schema=999)),
        ("wrong axes", lambda d: d.update(axes=["imag", "real", "magnitude"])),
        ("unknown caps", lambda d: d.update(caps={"kind": "logarithmic"})),
        ("unknown walls", lambda d: d["occluder"].update(walls={"kind": "spline"})),
        ("unknown region", lambda d: d["occluder"].update(region={"kind": "mask"})),
        ("missing domain", lambda d: d.pop("domain")),
    ]:
        d = json.loads(json.dumps(good))
        mutate(d)
        try:
            Manifest.from_dict(d)
        except BundleError:
            check(f"rejects {name}", True)
        else:
            check(f"rejects {name}", False, "accepted a malformed manifest")


def check_csr():
    """CSR and the running-integer tags are the same information."""
    rng = np.random.default_rng(5)
    lens = rng.integers(1, 6, size=40)          # includes 1-vertex runs, dropped
    xyz = rng.normal(size=(int(lens.sum()), 3))
    idx = np.repeat(np.arange(len(lens)) * 7, lens)   # non-contiguous tags
    verts, offsets = csr_from_indices(xyz, idx)
    back = indices_from_csr(offsets)
    kept = np.repeat(lens >= 2, lens)
    check("csr drops runs shorter than 2",
          len(verts) == int(lens[lens >= 2].sum()))
    check("csr preserves vertex order", np.array_equal(verts, xyz[kept]))
    check("csr offsets invert to contiguous tags",
          len(back) == len(verts) and np.all(np.diff(back) >= 0))


def check_axis_exchange():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(20, 3))
    check("swap_to_world is an involution",
          np.array_equal(swap_to_world(swap_to_world(a)), a))
    check("swap_to_world exchanges columns 0 and 1",
          np.array_equal(swap_to_world(a)[:, 0], a[:, 1]))


def check_affine():
    rng = np.random.default_rng(8)
    xy = rng.normal(size=(30, 3))
    t = Affine2(1.0, 0.0, 3.0, 0.0, -1.0, 2.0)
    out = t.apply(xy)
    check("Affine2 leaves z alone", np.array_equal(out[:, 2], xy[:, 2]))
    check("Affine2 round trips through JSON",
          Affine2.from_dict(t.to_dict()) == t)


def check_perimeter():
    """The polygon predicate must agree with the rectangle it describes."""
    # rectangle() is a set of walls in front/back/left/right order, not a
    # traversal, so it has no interior and must refuse to answer contains().
    walls = Perimeter.rectangle((0.0, 2.5), (-5.5, 4.0))
    try:
        walls.contains(1.0, 1.0)
    except ValueError:
        check("rectangle() refuses contains(): it is not a traversal", True)
    else:
        check("rectangle() refuses contains(): it is not a traversal", False,
              "answered for a perimeter with no interior")

    rng = np.random.default_rng(9)
    im = rng.uniform(-1, 3.5, 5000)
    re = rng.uniform(-7, 5, 5000)
    want = (im > 0) & (im < 2.5) & (re > -5.5) & (re < 4.0)
    # Perimeter.rectangle emits front/back/left/right, not a traversal, so
    # order the corners into a loop before testing containment.
    loop = Perimeter([Edge((0.0, -5.5), (0.0, 4.0)), Edge((0.0, 4.0), (2.5, 4.0)),
                      Edge((2.5, 4.0), (2.5, -5.5)), Edge((2.5, -5.5), (0.0, -5.5))])
    check("contains() agrees with the rectangle it outlines",
          np.array_equal(loop.contains(im, re), want))


def check_camera_fixtures():
    """The camera fixture must reproduce under the recorded parameters — the
    file says what it claims to say."""
    for path in sorted((FIXTURES / "camera").glob("*.json")):
        spec = json.loads(path.read_text())["plate"]
        proj = Projection(shear=spec["shear"], x_angle=spec["xAngle"],
                          z_angle=spec["zAngle"], flip_x=spec["flipX"],
                          y_scale=spec["yScale"])
        world = np.load(path.parent / f"{path.stem}.points.npy")
        want = np.load(path.parent / f"{path.stem}.projected.npy")
        got = proj.apply(world[:, [1, 0, 2]])
        err = float(np.abs(got - want).max())
        check(f"camera/{path.stem} reproduces", err == 0.0, f"max |Δ| = {err:g}")


def check_clip_fixture():
    """The stored expectation must be exactly what the stored inputs produce."""
    from kurven.outline import clip_hidden_lines
    from kurven.zbuffer import ZBuffer

    d = FIXTURES / "clip"
    meta = json.loads((d / "meta.json").read_text())
    depth = np.load(d / "depth.npy").astype(np.float64)
    view = np.load(d / "view.npy")
    offsets = np.load(d / "view.idx.npy")
    want = np.load(d / "expected.npy")
    want_off = np.load(d / "expected.idx.npy")

    zb = ZBuffer(meta["axis0"][0], meta["axis0"][1],
                 meta["axis1"][0], meta["axis1"][1], tuple(meta["shape"]))
    zb.buffer = depth
    segs = clip_hidden_lines(zb, view, indices_from_csr(offsets),
                             margin=meta["margin"])
    ok = len(segs) == len(want_off) - 1 == meta["segments"]
    if ok and segs:
        ok = np.array_equal(np.concatenate(segs), want)
    check("clip fixture reproduces exactly", bool(ok),
          f"{len(segs)} segments")


def main():
    print("bundle contract (python lane)")
    check_manifest_roundtrip()
    check_bundle_reads()
    check_schema_rejection()
    check_csr()
    check_axis_exchange()
    check_affine()
    check_perimeter()
    check_camera_fixtures()
    check_clip_fixture()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
