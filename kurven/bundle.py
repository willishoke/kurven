"""The `.kurven` bundle: the typed contract between the Python pipeline and the
Swift frontend.

The seam in kurven is not "frontend vs backend"; it is *camera-independent*
work (sample → contour → lift) versus *camera-dependent* work (project →
depth-buffer → clip → ink). Everything on the left of that line is expensive,
needs scipy, and does not change when you move the camera. Everything on the
right must run per frame. A bundle is the value that crosses the line: a
directory package holding a typed manifest plus `.npy` arrays.

    recip.kurven/
      manifest.json
      height.npy              float32 (ny, nx)   ny = imag, nx = real
      phase.npy               float32 (ny, nx)   optional
      layers/
        mag_major.npy         float64 (N, 3) world (x, y, z)
        mag_major.idx.npy     int64   (P + 1,)  CSR path offsets
      occluder/
        walls.npy, walls.tri.npy

World coordinates
-----------------
The rest of the library carries contour vertices as **(imag, real, z)** — a
convention inherited from the notebooks that has caused most of the historical
bugs in this codebase. The bundle does not inherit it. Bundle arrays are
**(x = real, y = imag, z = |f|)**, and `height.npy` is row-major with rows
indexing imag and columns indexing real. `swap_to_world` performs the exchange,
and it is the only place in the exporter that does.

`PlateProjection` records the *Python* `Projection` parameters verbatim, so a
consumer can rebuild the exact plate camera; because `Projection.apply` reads
column 0 as the sheared/flipped axis (imag) and column 1 as the scaled axis
(real), a consumer working in world order must fold the axis exchange into its
camera matrix. That is one matrix, built once, and it is what the camera
equivalence fixtures (`tests/fixtures/`) pin down.

Path identity
-------------
`Surface.lift_contours` tags each path with a running integer and downstream
code recovers paths with `itertools.groupby`. Integer tags need ad-hoc offset
arithmetic to keep unrelated runs from welding together. The bundle drops them:
a layer is CSR — a flat `(N, 3)` vertex array plus `(P + 1,)` offsets — so path
identity is structural.

Sum types are encoded as tagged objects, `{"kind": "...", ...}`, mirroring the
Swift `Codable` enums one-for-one. `Manifest.to_json` emits canonical JSON
(sorted keys, no insignificant whitespace) so a round trip through either
language is byte-comparable; that comparison is the whole schema test.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = 1

#: The one true axis convention for everything inside a bundle. Recorded in the
#: manifest so the file states it, rather than relying on a reader's memory.
AXES = ("real", "imag", "magnitude")


class BundleError(Exception):
    """Malformed bundle or manifest — always raised with the offending key."""


def _require(d, key, kind):
    if key not in d:
        raise BundleError(f"{kind}: missing key {key!r}")
    return d[key]


def _tag(d, kind):
    return _require(d, "kind", kind)


# --------------------------------------------------------------------------
# leaves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def to_dict(self):
        return {"lo": float(self.lo), "hi": float(self.hi)}

    @classmethod
    def from_dict(cls, d):
        return cls(float(_require(d, "lo", "Interval")),
                   float(_require(d, "hi", "Interval")))


@dataclass(frozen=True)
class Domain:
    """The sampled rectangle of the complex plane, in world axis order."""

    real: Interval
    imag: Interval

    def to_dict(self):
        return {"real": self.real.to_dict(), "imag": self.imag.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(Interval.from_dict(_require(d, "real", "Domain")),
                   Interval.from_dict(_require(d, "imag", "Domain")))


@dataclass(frozen=True)
class GridRef:
    """A 2D array on disk. `shape` is (ny, nx) = (n_imag, n_real)."""

    file: str
    shape: tuple[int, int]
    dtype: str

    def to_dict(self):
        return {"file": self.file, "shape": [int(self.shape[0]), int(self.shape[1])],
                "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d):
        s = _require(d, "shape", "GridRef")
        if len(s) != 2:
            raise BundleError(f"GridRef: shape must have rank 2, got {s!r}")
        return cls(str(_require(d, "file", "GridRef")), (int(s[0]), int(s[1])),
                   str(_require(d, "dtype", "GridRef")))


@dataclass(frozen=True)
class Affine2:
    """A 2×3 affine map on the world (x, y) plane:

        x' = a·x + b·y + tx
        y' = c·x + d·y + ty

    The elliptic plate's 18 reflected copies are exactly 18 of these, and so is
    the identity for the base tile. Stored as a flat row-major 6-list.
    """

    a: float
    b: float
    tx: float
    c: float
    d: float
    ty: float

    @classmethod
    def identity(cls):
        return cls(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @classmethod
    def scale_offset(cls, sx, sy, tx, ty):
        """The only family the examples use: axis-aligned reflect-and-translate."""
        return cls(float(sx), 0.0, float(tx), 0.0, float(sy), float(ty))

    def apply(self, xy):
        """Map an (N, 2) or (N, 3) world array; a z column is passed through."""
        xy = np.asarray(xy, dtype=float)
        out = xy.copy()
        x, y = xy[:, 0], xy[:, 1]
        out[:, 0] = self.a * x + self.b * y + self.tx
        out[:, 1] = self.c * x + self.d * y + self.ty
        return out

    def to_dict(self):
        return {"m": [float(self.a), float(self.b), float(self.tx),
                      float(self.c), float(self.d), float(self.ty)]}

    @classmethod
    def from_dict(cls, d):
        m = _require(d, "m", "Affine2")
        if len(m) != 6:
            raise BundleError(f"Affine2: expected 6 coefficients, got {len(m)}")
        return cls(*(float(v) for v in m))


@dataclass(frozen=True)
class Edge:
    """A straight perimeter segment in world (x, y), sampled at `density`
    points. Mirrors `kurven.perimeter.Edge` with the axes exchanged."""

    start: tuple[float, float]
    end: tuple[float, float]
    density: int

    def to_dict(self):
        return {"start": [float(self.start[0]), float(self.start[1])],
                "end": [float(self.end[0]), float(self.end[1])],
                "density": int(self.density)}

    @classmethod
    def from_dict(cls, d):
        s = _require(d, "start", "Edge")
        e = _require(d, "end", "Edge")
        return cls((float(s[0]), float(s[1])), (float(e[0]), float(e[1])),
                   int(_require(d, "density", "Edge")))


@dataclass(frozen=True)
class Perimeter:
    """An ordered boundary outline in world coordinates — the description from
    which cut-face walls are derived on the consumer side."""

    edges: tuple[Edge, ...]

    def contains(self, x, y):
        """Even-odd point-in-polygon in world `(x, y)`. Broadcasts."""
        return point_in_polygon(x, y, [e.start for e in self.edges])

    def to_dict(self):
        return {"edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(Edge.from_dict(e) for e in _require(d, "edges", "Perimeter")))


# --------------------------------------------------------------------------
# caps: the sum type that owns z-clamping
# --------------------------------------------------------------------------


class Caps:
    """How the surface is truncated from above.

    A cap is a property of the *model*, not the camera: it is what makes a pole
    spire a visible plateau instead of a needle. `Projection.z_clamp` (zeta)
    conflates the two; the exporter moves it here.
    """

    @staticmethod
    def from_dict(d):
        kind = _tag(d, "Caps")
        if kind == "none":
            return NoCaps()
        if kind == "uniform":
            return UniformCap(float(_require(d, "z", "Caps.uniform")))
        if kind == "realBands":
            return RealBandCaps(tuple(
                RealBand.from_dict(b) for b in _require(d, "bands", "Caps.realBands")))
        raise BundleError(f"Caps: unknown kind {kind!r}")


@dataclass(frozen=True)
class NoCaps(Caps):
    def to_dict(self):
        return {"kind": "none"}

    def apply(self, mag, x=None):
        return mag


@dataclass(frozen=True)
class UniformCap(Caps):
    z: float

    def to_dict(self):
        return {"kind": "uniform", "z": float(self.z)}

    def apply(self, mag, x=None):
        return np.minimum(self.z, mag)


@dataclass(frozen=True)
class RealBand:
    """`cap` applies where x (= Re z) is below `below`; bands are tested in
    order and the first match wins, so they read as a staircase in Re."""

    below: float
    cap: float

    def to_dict(self):
        return {"below": float(self.below), "cap": float(self.cap)}

    @classmethod
    def from_dict(cls, d):
        return cls(float(_require(d, "below", "RealBand")),
                   float(_require(d, "cap", "RealBand")))


@dataclass(frozen=True)
class RealBandCaps(Caps):
    """Gamma's per-spire truncation: the cap height depends on Re(z)."""

    bands: tuple[RealBand, ...]

    def to_dict(self):
        return {"kind": "realBands", "bands": [b.to_dict() for b in self.bands]}

    def apply(self, mag, x):
        x = np.asarray(x, dtype=float)
        cap = np.full(np.shape(x), np.inf)
        assigned = np.zeros(np.shape(x), dtype=bool)
        for band in self.bands:
            hit = (~assigned) & (x < band.below)
            cap = np.where(hit, band.cap, cap)
            assigned |= hit
        return np.minimum(cap, mag)


# --------------------------------------------------------------------------
# occluder
# --------------------------------------------------------------------------


class Walls:
    """The vertical cut-face curtains, either dumped or described.

    Dumping them (`WallMesh`) keeps the wall crests at the *analytic* surface
    height, which is what the Python plates draw; deriving them (`WallPerimeter`)
    re-evaluates the crest by bilinear lookup on `height.npy` and so differs by
    the grid's interpolation error. Phase 1 exports the mesh; Phase 3 exports
    the perimeter and the difference becomes a measurable quantity.
    """

    @staticmethod
    def from_dict(d):
        kind = _tag(d, "Walls")
        if kind == "none":
            return NoWalls()
        if kind == "mesh":
            return WallMesh(str(_require(d, "vertices", "Walls.mesh")),
                            str(_require(d, "triangles", "Walls.mesh")))
        if kind == "perimeter":
            return WallPerimeter(
                Perimeter.from_dict(_require(d, "perimeter", "Walls.perimeter")),
                float(_require(d, "base", "Walls.perimeter")))
        raise BundleError(f"Walls: unknown kind {kind!r}")


@dataclass(frozen=True)
class NoWalls(Walls):
    def to_dict(self):
        return {"kind": "none"}


@dataclass(frozen=True)
class WallMesh(Walls):
    vertices: str
    triangles: str

    def to_dict(self):
        return {"kind": "mesh", "vertices": self.vertices, "triangles": self.triangles}


@dataclass(frozen=True)
class WallPerimeter(Walls):
    perimeter: Perimeter
    base: float = 0.0

    def to_dict(self):
        return {"kind": "perimeter", "perimeter": self.perimeter.to_dict(),
                "base": float(self.base)}


class Region:
    """The footprint the heightfield is rasterized over.

    zeta's landscape is a staircase, not a rectangle: the notch in front of the
    s = 1 pole is where the plate cuts away to show the cross-section. The notch
    is not a mask applied afterwards -- it is an absence of geometry, so a cell
    contributes triangles only when all four of its corners are inside. Carrying
    it as a polygon rather than as a second mask array is what lets the same
    outline drive the walls, the ground ink and the mesh.
    """

    @staticmethod
    def from_dict(d):
        kind = _tag(d, "Region")
        if kind == "full":
            return FullRegion()
        if kind == "inside":
            return InsideRegion(
                Perimeter.from_dict(_require(d, "perimeter", "Region.inside")))
        raise BundleError(f"Region: unknown kind {kind!r}")


@dataclass(frozen=True)
class FullRegion(Region):
    def to_dict(self):
        return {"kind": "full"}


@dataclass(frozen=True)
class InsideRegion(Region):
    perimeter: Perimeter

    def to_dict(self):
        return {"kind": "inside", "perimeter": self.perimeter.to_dict()}


@dataclass(frozen=True)
class Occluder:
    """What blocks sight lines.

    The heightfield is never dumped as triangles. It is `height.npy` plus a
    `step`, instanced once per entry in `tiles`; a consumer rasterizes it from
    the texture at whatever level of detail it is drawing at. (A single explicit
    mesh for the elliptic plate would be six million triangles on disk to say
    what one affine list already says.) Only the walls are geometry, and they
    are a few thousand triangles.

    `tiles` always contains the identity as its first entry, and `region` is the
    footprint each tile is clipped to.
    """

    step: int
    tiles: tuple[Affine2, ...]
    walls: Walls
    region: Region = field(default_factory=lambda: FullRegion())
    base: float = 0.0

    def to_dict(self):
        return {"step": int(self.step),
                "tiles": [t.to_dict() for t in self.tiles],
                "walls": self.walls.to_dict(),
                "region": self.region.to_dict(),
                "base": float(self.base)}

    @classmethod
    def from_dict(cls, d):
        return cls(int(_require(d, "step", "Occluder")),
                   tuple(Affine2.from_dict(t) for t in _require(d, "tiles", "Occluder")),
                   Walls.from_dict(_require(d, "walls", "Occluder")),
                   Region.from_dict(_require(d, "region", "Occluder")),
                   float(d.get("base", 0.0)))


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------

LAYER_ROLES = ("magnitude", "phase", "scaffold", "outline")
HEIGHT_POLICIES = ("surface", "level", "magnitude")
CONTOUR_FIELDS = ("magnitude", "phase")
KEEP_AXES = ("real", "imag")


class Keep:
    """Which vertices of a derived contour survive.

    A small, closed vocabulary rather than an expression language. Everything
    the plates actually ask for is here -- stay inside the occluder's footprint,
    stay under the cap, stay within a band of one axis -- and a consumer can
    evaluate all of it without an interpreter. Anything a function needs beyond
    this is a sign the layer should be dumped rather than described, which is
    what `LayerFile` is for.
    """

    @staticmethod
    def from_dict(d):
        kind = _tag(d, "Keep")
        if kind == "all":
            return KeepAll()
        if kind == "region":
            return KeepRegion()
        if kind == "belowCap":
            return KeepBelowCap()
        if kind == "band":
            axis = str(_require(d, "axis", "Keep.band"))
            if axis not in KEEP_AXES:
                raise BundleError(f"Keep.band: unknown axis {axis!r}")
            return KeepBand(axis, float(_require(d, "lo", "Keep.band")),
                            float(_require(d, "hi", "Keep.band")))
        if kind == "every":
            return KeepEvery(tuple(Keep.from_dict(k)
                                   for k in _require(d, "of", "Keep.every")))
        raise BundleError(f"Keep: unknown kind {kind!r}")


@dataclass(frozen=True)
class KeepAll(Keep):
    def to_dict(self):
        return {"kind": "all"}


@dataclass(frozen=True)
class KeepRegion(Keep):
    """Inside `occluder.region` -- the same polygon the heightfield is cut to,
    referenced rather than repeated."""

    def to_dict(self):
        return {"kind": "region"}


@dataclass(frozen=True)
class KeepBelowCap(Keep):
    def to_dict(self):
        return {"kind": "belowCap"}


@dataclass(frozen=True)
class KeepBand(Keep):
    axis: str
    lo: float
    hi: float

    def to_dict(self):
        return {"kind": "band", "axis": self.axis,
                "lo": float(self.lo), "hi": float(self.hi)}


@dataclass(frozen=True)
class KeepEvery(Keep):
    of: tuple

    def to_dict(self):
        return {"kind": "every", "of": [k.to_dict() for k in self.of]}


class LayerSource:
    """Where a layer's geometry comes from: a file, or a description.

    A dumped layer is the answer; a described one is the question. Describing it
    is what lets a consumer move the levels and get a new answer, and it is what
    shrinks a bundle from megabytes of vertices to a list of numbers -- zeta's
    contour layers are 3.4 MB dumped and four lines described.

    Not every layer can be described. Elliptic's phase contours are trimmed by a
    lattice-intersection test written for that plate and nothing else; layers
    like that stay files, and the honest thing is for the schema to say so
    rather than to grow an expression language to accommodate one example.
    """

    @staticmethod
    def from_dict(d):
        kind = _tag(d, "LayerSource")
        if kind == "file":
            return LayerFile(str(_require(d, "vertices", "LayerSource.file")),
                             str(_require(d, "offsets", "LayerSource.file")))
        if kind == "contour":
            field = str(_require(d, "field", "LayerSource.contour"))
            if field not in CONTOUR_FIELDS:
                raise BundleError(f"LayerSource.contour: unknown field {field!r}")
            return LayerContour(
                field,
                tuple(float(v) for v in _require(d, "levels", "LayerSource.contour")),
                Keep.from_dict(_require(d, "keep", "LayerSource.contour")),
                bool(d.get("tiled", False)))
        raise BundleError(f"LayerSource: unknown kind {kind!r}")


@dataclass(frozen=True)
class LayerFile(LayerSource):
    vertices: str
    offsets: str

    def to_dict(self):
        return {"kind": "file", "vertices": self.vertices, "offsets": self.offsets}


@dataclass(frozen=True)
class LayerContour(LayerSource):
    """Iso-lines of `field` at `levels`, lifted by the layer's height policy,
    filtered by `keep`, and replicated once per occluder tile when `tiled`."""

    field: str
    levels: tuple
    keep: Keep = field(default_factory=lambda: KeepAll())
    tiled: bool = False

    def to_dict(self):
        return {"kind": "contour", "field": self.field,
                "levels": [float(v) for v in self.levels],
                "keep": self.keep.to_dict(),
                "tiled": bool(self.tiled)}


@dataclass(frozen=True)
class LayerSpec:
    """One stratum of ink: a CSR polyline set plus how it is drawn.

    `width` is a matplotlib line width in points, carried through to the SVG
    stroke width; a realtime preview that cannot draw wide lines is free to
    ignore it.

    `clipped` is false for ink that is drawn *on* the occluding geometry rather
    than draped over it — a cut-face hatch lies in the wall it hatches, so a
    depth test would erase roughly half of it to no purpose.
    """

    name: str
    role: str
    source: LayerSource
    width: float
    height_policy: str
    color: str = "#000000"
    clipped: bool = True

    def __post_init__(self):
        if self.role not in LAYER_ROLES:
            raise BundleError(f"LayerSpec {self.name!r}: unknown role {self.role!r}")
        if self.height_policy not in HEIGHT_POLICIES:
            raise BundleError(
                f"LayerSpec {self.name!r}: unknown height policy {self.height_policy!r}")

    @property
    def files(self):
        """`(vertices, offsets)` when this layer is dumped, else `None`."""
        return ((self.source.vertices, self.source.offsets)
                if isinstance(self.source, LayerFile) else None)

    def to_dict(self):
        return {"name": self.name, "role": self.role, "source": self.source.to_dict(),
                "width": float(self.width),
                "heightPolicy": self.height_policy, "color": self.color,
                "clipped": bool(self.clipped)}

    @classmethod
    def from_dict(cls, d):
        return cls(str(_require(d, "name", "LayerSpec")),
                   str(_require(d, "role", "LayerSpec")),
                   LayerSource.from_dict(_require(d, "source", "LayerSpec")),
                   float(_require(d, "width", "LayerSpec")),
                   str(_require(d, "heightPolicy", "LayerSpec")),
                   str(d.get("color", "#000000")),
                   bool(d.get("clipped", True)))


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateProjection:
    """`kurven.projection.Projection`'s parameters, verbatim.

    Column 0 of the array `Projection.apply` consumes is imag and column 1 is
    real, so `flip_x` flips imag and `y_scale` scales real. A consumer holding
    world-order data folds the axis exchange into its camera matrix; the
    parameters themselves are copied across untouched so the two cameras can be
    compared numerically rather than by inspection.
    """

    shear: float
    x_angle: float
    z_angle: float
    flip_x: bool
    y_scale: float | None

    def to_dict(self):
        return {"shear": float(self.shear), "xAngle": float(self.x_angle),
                "zAngle": float(self.z_angle), "flipX": bool(self.flip_x),
                "yScale": None if self.y_scale is None else float(self.y_scale)}

    @classmethod
    def from_dict(cls, d):
        ys = d.get("yScale")
        return cls(float(_require(d, "shear", "PlateProjection")),
                   float(_require(d, "xAngle", "PlateProjection")),
                   float(_require(d, "zAngle", "PlateProjection")),
                   bool(_require(d, "flipX", "PlateProjection")),
                   None if ys is None else float(ys))

    @classmethod
    def of(cls, projection):
        """Read the parameters back off a live `kurven.projection.Projection`."""
        return cls(projection.shear, projection.x_angle, projection.z_angle,
                   projection.flip_x, projection.y_scale)


@dataclass(frozen=True)
class CameraPreset:
    """A named camera, with the plate's own depth resolution and clip margin —
    the settings under which the published plate was made."""

    name: str
    plate: PlateProjection
    margin: float
    buffer: int

    def to_dict(self):
        return {"name": self.name, "plate": self.plate.to_dict(),
                "margin": float(self.margin), "buffer": int(self.buffer)}

    @classmethod
    def from_dict(cls, d):
        return cls(str(_require(d, "name", "CameraPreset")),
                   PlateProjection.from_dict(_require(d, "plate", "CameraPreset")),
                   float(_require(d, "margin", "CameraPreset")),
                   int(_require(d, "buffer", "CameraPreset")))


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


@dataclass(frozen=True)
class Provenance:
    """Enough to regenerate the bundle. `cpu_count` is here because contourpy's
    threaded backend stitches chunk seams in completion order: a bundle written
    with more than one chunk is not reproducible, and a reader comparing two
    bundles needs to know that before it blames the diff on anything else."""

    function: str
    params: dict
    cpu_count: int
    #: The `kurven.export` example that produced this, when one did. It is what
    #: lets a consumer ask the service for the *same* landscape at a different
    #: resolution -- `function` names the mathematics, this names the recipe.
    example: str = ""
    git_sha: str = field(default_factory=git_sha)

    def to_dict(self):
        return {"function": self.function, "example": self.example,
                "params": {k: self.params[k] for k in sorted(self.params)},
                "cpuCount": int(self.cpu_count), "gitSha": self.git_sha}

    @classmethod
    def from_dict(cls, d):
        return cls(str(_require(d, "function", "Provenance")),
                   dict(_require(d, "params", "Provenance")),
                   int(_require(d, "cpuCount", "Provenance")),
                   str(d.get("example", "")),
                   str(_require(d, "gitSha", "Provenance")))


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    schema: int
    axes: tuple[str, str, str]
    domain: Domain
    height: GridRef
    phase: GridRef | None
    caps: Caps
    occluder: Occluder
    layers: tuple[LayerSpec, ...]
    presets: tuple[CameraPreset, ...]
    provenance: Provenance

    def to_dict(self):
        return {
            "schema": int(self.schema),
            "axes": list(self.axes),
            "domain": self.domain.to_dict(),
            "height": self.height.to_dict(),
            "phase": None if self.phase is None else self.phase.to_dict(),
            "caps": self.caps.to_dict(),
            "occluder": self.occluder.to_dict(),
            "layers": [l.to_dict() for l in self.layers],
            "presets": [p.to_dict() for p in self.presets],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        schema = int(_require(d, "schema", "Manifest"))
        if schema != SCHEMA:
            raise BundleError(f"Manifest: schema {schema} != supported {SCHEMA}")
        axes = tuple(str(a) for a in _require(d, "axes", "Manifest"))
        if axes != AXES:
            raise BundleError(f"Manifest: axes {axes} != {AXES}")
        ph = d.get("phase")
        return cls(
            schema, axes,
            Domain.from_dict(_require(d, "domain", "Manifest")),
            GridRef.from_dict(_require(d, "height", "Manifest")),
            None if ph is None else GridRef.from_dict(ph),
            Caps.from_dict(_require(d, "caps", "Manifest")),
            Occluder.from_dict(_require(d, "occluder", "Manifest")),
            tuple(LayerSpec.from_dict(x) for x in _require(d, "layers", "Manifest")),
            tuple(CameraPreset.from_dict(x) for x in _require(d, "presets", "Manifest")),
            Provenance.from_dict(_require(d, "provenance", "Manifest")),
        )

    def to_json(self):
        """Canonical JSON: sorted keys, tight separators, trailing newline. Two
        manifests are equal iff their canonical JSON is byte-equal, which is
        what the cross-language contract test compares."""
        return json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def preset(self, name):
        for p in self.presets:
            if p.name == name:
                return p
        raise BundleError(
            f"Manifest: no preset {name!r} (have {[p.name for p in self.presets]})")


# --------------------------------------------------------------------------
# world-order conversion
# --------------------------------------------------------------------------


def point_in_polygon(x, y, corners):
    """Even-odd ray crossing, scanning along **world x (= real)**. Broadcasts.

    The single definition of "inside" in kurven. Both `Perimeter` types call it
    -- `kurven.perimeter.Perimeter` by exchanging its `(imag, real)` corners
    first -- so the mask the occluder mesh is cut with, the predicate the
    contours are filtered by, and whatever a consumer computes from the manifest
    are the same function of the same corners, down to the boundary cases where
    a scan along the other axis would disagree.

    `corners` is a closed polygon in world order; the last corner joins the
    first.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    P = np.asarray(corners, dtype=float)
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        if a[0] == b[0]:
            continue
        crossing = (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]) + a[1]
        inside ^= ((a[0] > x) != (b[0] > x)) & (y < crossing)
    return inside


def swap_to_world(imrez):
    """(imag, real, z) → (x = real, y = imag, z). The single axis exchange."""
    a = np.asarray(imrez, dtype=float)
    out = np.empty_like(a)
    out[:, 0] = a[:, 1]
    out[:, 1] = a[:, 0]
    if a.shape[1] > 2:
        out[:, 2:] = a[:, 2:]
    return out


def swap_from_world(xyz):
    """(x = real, y = imag, z) → (imag, real, z). Its own inverse, named twice
    so call sites read in the direction they mean."""
    return swap_to_world(xyz)


def csr_from_indices(xyz, indices):
    """Turn `(xyz, indices)` from `Surface.lift_contours` into CSR.

    `indices` tags each vertex with its path; runs are contiguous. Returns
    `(vertices, offsets)` where path `p` is `vertices[offsets[p]:offsets[p+1]]`.
    Runs shorter than two vertices are dropped — they draw nothing, and keeping
    them would let a consumer emit a degenerate stroke.
    """
    xyz = np.asarray(xyz, dtype=float)
    indices = np.asarray(indices)
    if len(xyz) == 0:
        return np.zeros((0, 3)), np.zeros(1, dtype=np.int64)
    if len(xyz) != len(indices):
        raise BundleError(f"csr: {len(xyz)} vertices vs {len(indices)} indices")

    breaks = np.flatnonzero(indices[1:] != indices[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(xyz)]])

    keep = (ends - starts) >= 2
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return np.zeros((0, 3)), np.zeros(1, dtype=np.int64)

    verts = np.concatenate([xyz[s:e] for s, e in zip(starts, ends)])
    offsets = np.concatenate([[0], np.cumsum(ends - starts)]).astype(np.int64)
    return verts, offsets


def indices_from_csr(offsets):
    """The inverse of `csr_from_indices`' index column: a per-vertex path tag,
    for handing a CSR layer back to `clip_hidden_lines`."""
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts)


# --------------------------------------------------------------------------
# reading and writing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    """A decoded layer: CSR vertices in world coordinates, plus its spec."""

    spec: LayerSpec
    vertices: np.ndarray | None
    offsets: np.ndarray | None

    @property
    def derived(self):
        return self.vertices is None

    @property
    def paths(self):
        if self.vertices is None:
            return []
        return [self.vertices[self.offsets[i]:self.offsets[i + 1]]
                for i in range(len(self.offsets) - 1)]


@dataclass(frozen=True)
class LoadedBundle:
    """A bundle read back off disk. Arrays are eager: they are the point."""

    path: Path
    manifest: Manifest
    height: np.ndarray
    phase: np.ndarray | None
    layers: tuple[Layer, ...]
    wall_vertices: np.ndarray | None
    wall_triangles: np.ndarray | None

    def layer(self, name):
        for l in self.layers:
            if l.spec.name == name:
                return l
        raise BundleError(
            f"bundle: no layer {name!r} (have {[l.spec.name for l in self.layers]})")


def write_bundle(path, *, manifest, height, phase=None, layers, walls=None):
    """Write a bundle directory.

    `layers` maps a `LayerSpec.name` to `(vertices, offsets)` in world order;
    `walls` is `(vertices, triangles)` in world order when the manifest names a
    `WallMesh`. The manifest is written last, so a bundle whose `manifest.json`
    exists is a bundle whose arrays are complete.
    """
    path = Path(path)
    (path / "layers").mkdir(parents=True, exist_ok=True)

    np.save(path / manifest.height.file, np.ascontiguousarray(height, dtype=np.float32))
    if manifest.phase is not None:
        if phase is None:
            raise BundleError("manifest declares a phase grid but none was given")
        np.save(path / manifest.phase.file, np.ascontiguousarray(phase, dtype=np.float32))

    for spec in manifest.layers:
        files = spec.files
        if files is None:
            continue                    # described, not dumped
        if spec.name not in layers:
            raise BundleError(f"manifest declares layer {spec.name!r} but none was given")
        verts, offsets = layers[spec.name]
        np.save(path / files[0], np.ascontiguousarray(verts, dtype=np.float64))
        np.save(path / files[1], np.ascontiguousarray(offsets, dtype=np.int64))

    w = manifest.occluder.walls
    if isinstance(w, WallMesh):
        if walls is None:
            raise BundleError("manifest declares a wall mesh but none was given")
        (path / "occluder").mkdir(parents=True, exist_ok=True)
        wv, wt = walls
        np.save(path / w.vertices, np.ascontiguousarray(wv, dtype=np.float64))
        np.save(path / w.triangles, np.ascontiguousarray(wt, dtype=np.int64))

    (path / "manifest.json").write_text(manifest.to_json())
    return path


def read_bundle(path):
    """Read a bundle directory back into values. Shapes declared in the manifest
    are checked against the arrays; a mismatch is a `BundleError`, not a
    surprise three stages later."""
    path = Path(path)
    mf = path / "manifest.json"
    if not mf.exists():
        raise BundleError(f"bundle: no manifest at {mf}")
    manifest = Manifest.from_json(mf.read_text())

    def grid(ref):
        arr = np.load(path / ref.file)
        if arr.shape != tuple(ref.shape):
            raise BundleError(
                f"bundle: {ref.file} has shape {arr.shape}, manifest says {tuple(ref.shape)}")
        return arr

    height = grid(manifest.height)
    phase = grid(manifest.phase) if manifest.phase is not None else None

    layers = []
    for spec in manifest.layers:
        files = spec.files
        if files is None:
            # A described layer carries no arrays; deriving it is the reader's
            # job, and this reader is the Python one, which already has the
            # pipeline that made it.
            layers.append(Layer(spec, None, None))
            continue
        verts = np.load(path / files[0])
        offsets = np.load(path / files[1])
        if len(offsets) and offsets[-1] != len(verts):
            raise BundleError(
                f"bundle: layer {spec.name!r} offsets end at {offsets[-1]}, "
                f"{len(verts)} vertices")
        layers.append(Layer(spec, verts, offsets))

    wv = wt = None
    w = manifest.occluder.walls
    if isinstance(w, WallMesh):
        wv = np.load(path / w.vertices)
        wt = np.load(path / w.triangles)

    return LoadedBundle(path, manifest, height, phase, tuple(layers), wv, wt)


def deterministic_chunk_count():
    """1, always — and a reminder why. Exporters pass this to `contour_levels`
    so a bundle is a reproducible value rather than a sample from a race."""
    return 1
