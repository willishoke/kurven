"""`Scene`: the camera-independent half of a plate.

An example used to be one `main()` that sampled, contoured, lifted, projected,
rasterized, clipped and drew. Only the first three of those stages depend on the
function; the rest depend on where you put the camera. Splitting the examples at
that seam gives each one a `build_scene()` returning this value — everything a
plate is before anyone decides how to look at it — and a `render_plate()` that
takes the scene and a `Projection`.

The split is what makes a `.kurven` bundle possible: a bundle *is* a `Scene`,
serialized (`kurven.export`). It is also what makes the split verifiable, since
`main()` is now literally the composition of the two halves and must produce the
same SVG it always did (`tests/verify_refactor.py`).

Arrays here are still in the library's `(imag, real, z)` column order; the
exchange to world order happens once, in `kurven.export`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kurven.bundle import Affine2, CameraPreset, Caps, NoCaps, Perimeter


@dataclass(frozen=True)
class InkLayer:
    """One stratum of drawn line-work, lifted to 3D but not yet projected.

    `xyz` is `(N, 3)` in `(imag, real, z)`; `indices` tags each vertex with its
    path exactly as `Surface.lift_contours` does, because that is what
    `clip_hidden_lines` consumes. `height_policy` records how z was chosen so a
    consumer that re-lifts the layer itself (Phase 3, native contouring) can
    reproduce it. `clipped` is false for ink that lies *in* the occluding
    geometry (a cut-face hatch), which a depth test would half erase.
    """

    name: str
    role: str
    xyz: np.ndarray
    indices: np.ndarray
    width: float
    height_policy: str = "surface"
    color: str = "#000000"
    clipped: bool = True

    def runs(self):
        """`(start, stop)` slices of each path — the same contiguous runs of
        equal `indices` that `clip_hidden_lines` groups by, recovered without
        `groupby` so a caller can slice a *projected* copy of `xyz` the same
        way."""
        if len(self.indices) == 0:
            return []
        breaks = np.flatnonzero(self.indices[1:] != self.indices[:-1]) + 1
        starts = np.concatenate([[0], breaks])
        stops = np.concatenate([breaks, [len(self.indices)]])
        return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= 2]

    def split(self, values):
        """Slice a per-vertex array (`xyz`, or a projected copy of it) into one
        array per path."""
        return [np.asarray(values)[a:b] for a, b in self.runs()]

    @classmethod
    def from_segments(cls, name, role, segments, width, **kw):
        """Build a layer from a list of independent polylines — the shape the
        scaffold generators return."""
        segments = [np.asarray(s, dtype=float) for s in segments]
        segments = [s for s in segments if len(s) >= 2]
        if not segments:
            return cls(name, role, np.zeros((0, 3)), np.zeros(0, dtype=np.int64),
                       width, **kw)
        xyz = np.concatenate(segments)
        indices = np.concatenate([np.full(len(s), i, dtype=np.int64)
                                  for i, s in enumerate(segments)])
        return cls(name, role, xyz, indices, width, **kw)


@dataclass(frozen=True)
class Scene:
    """Everything about a plate that survives moving the camera.

    `tiles` and `perimeter` are in **world** `(x = real, y = imag)` order — they
    are descriptions destined for the manifest, not arrays the Python pipeline
    indexes, so there is no reason to carry them in the legacy order. `walls` is
    mesh data and stays in `(imag, real, z)` like every other mesh here.
    """

    function: str
    params: dict
    surface: object                      # kurven.surface.Surface
    layers: tuple[InkLayer, ...]
    preset: CameraPreset
    caps: Caps = field(default_factory=NoCaps)
    occluder_step: int = 1
    tiles: tuple[Affine2, ...] = (Affine2.identity(),)
    perimeter: Perimeter | None = None
    walls: tuple = ()                    # ((vertices, triangles), ...)
    wall_base: float = 0.0

    def layer(self, name):
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(f"scene has no layer {name!r} "
                       f"(have {[l.name for l in self.layers]})")

    def ink(self, *roles):
        """Layers filtered by role, in declaration order."""
        return tuple(l for l in self.layers if not roles or l.role in roles)


def legacy_tile_transforms(tiles):
    """World-order `Affine2` tiles as `(imag, real, z)` vertex maps, for
    `occluder.build_occluder`.

    The identity is dropped: `build_occluder` always includes the untransformed
    base tile, so passing it again would rasterize the base twice. This is the
    one adapter between the bundle's world order and the library's legacy order
    on the mesh path.
    """
    out = []
    for t in tiles:
        if t == Affine2.identity():
            continue

        def xform(V, t=t):
            V = np.asarray(V, dtype=float)
            d = V.copy()
            d[:, 0] = t.c * V[:, 1] + t.d * V[:, 0] + t.ty     # imag
            d[:, 1] = t.a * V[:, 1] + t.b * V[:, 0] + t.tx     # real
            return d

        out.append(xform)
    return out
