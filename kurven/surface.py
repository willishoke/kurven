"""Surface: the sampled |f| / arg(f) landscape, analytic or cached.

A `Surface` is the single source of truth for everything that depends on the
function being visualized: the magnitude/phase grids the contours come from,
the heights the cut-face wall curtains rise to, and the heightfield mesh the
Z-buffer occluder is built from. It hides whether f is evaluated analytically
(exact point lookups) or loaded from a precomputed cache (nearest-pixel
lookups), so the rest of the pipeline never branches on that.

Conventions (shared with the examples):
  - `real` indexes axis 0, `imag` indexes axis 1; `values[a, b] = f(real[a] +
    1j*imag[b])`.
  - `grid_mesh` emits vertices as (imag, real, z) to match the projection.
"""

import numpy as np

from kurven.zbuffer import surface_grid_mesh


class Surface:
    def __init__(self, real, imag, values, *, z_limit=None, evaluator=None):
        self.real = np.asarray(real, dtype=float)
        self.imag = np.asarray(imag, dtype=float)
        self.values = np.asarray(values)
        if self.values.shape != (len(self.real), len(self.imag)):
            raise ValueError(
                f"values shape {self.values.shape} disagrees with "
                f"(len(real)={len(self.real)}, len(imag)={len(self.imag)})")
        self.mag = np.abs(self.values)
        self.angle = np.angle(self.values)
        self.z_limit = z_limit
        self._evaluator = evaluator

    @classmethod
    def from_function(cls, f, real, imag, *, z_limit=None):
        """Sample f on the real x imag grid, retaining f for exact point eval.

        `f` takes a complex array and returns a complex array.
        """
        real = np.asarray(real, dtype=float)
        imag = np.asarray(imag, dtype=float)
        grid = real[:, None] + 1j * imag[None, :]
        return cls(real, imag, f(grid), z_limit=z_limit, evaluator=f)

    @classmethod
    def from_cache(cls, values, real_bounds, imag_bounds, *, z_limit=None):
        """Wrap a precomputed (n_real, n_imag) complex array. No exact eval, so
        `mag_at` falls back to nearest-pixel lookup."""
        values = np.asarray(values)
        n_real, n_imag = values.shape
        real = np.linspace(real_bounds[0], real_bounds[1], n_real)
        imag = np.linspace(imag_bounds[0], imag_bounds[1], n_imag)
        return cls(real, imag, values, z_limit=z_limit)

    @property
    def clamped(self):
        """Magnitude clamped at `z_limit` — the surface the occluder meshes."""
        if self.z_limit is None:
            return self.mag
        return np.minimum(self.z_limit, self.mag)

    def mag_at(self, re, im):
        """|f| at arbitrary (re, im). Exact when analytic, else nearest-pixel.

        Broadcasts: `re` and `im` may be scalars or arrays.
        """
        re = np.asarray(re, dtype=float)
        im = np.asarray(im, dtype=float)
        if self._evaluator is not None:
            return np.abs(self._evaluator(re + 1j * im))
        r_idx = np.clip(np.rint(
            (re - self.real[0]) / (self.real[-1] - self.real[0]) * (len(self.real) - 1)
        ).astype(np.int64), 0, len(self.real) - 1)
        i_idx = np.clip(np.rint(
            (im - self.imag[0]) / (self.imag[-1] - self.imag[0]) * (len(self.imag) - 1)
        ).astype(np.int64), 0, len(self.imag) - 1)
        return self.mag[r_idx, i_idx]

    def height_at(self, re, im):
        """`mag_at` clamped at `z_limit` — the height a wall/occluder rises to."""
        h = self.mag_at(re, im)
        return h if self.z_limit is None else np.minimum(self.z_limit, h)

    def lift_contours(self, level_paths, *, start=0, height="surface", keep=None):
        """Lift contour level-paths to indexed 3D polylines, tagging each path
        with a running index — the shared 'assemble contour data' loop the
        examples all wrote by hand.

        `level_paths`: ordered iterable of `(level, [ (N, 2) xy arrays ])` as
        `contours.contour_levels` returns (col 0 = imag, col 1 = real).

        `height` sets each vertex's z:
            "surface"   z = self.height_at(re, im)  — clamped magnitude, the
                        height the contour lies on (magnitude OR phase contours)
            "level"     z = the contour level       — a magnitude isocontour
                        sits at exactly its level
            "magnitude" z = self.mag_at(re, im)     — unclamped magnitude

        `keep`: optional `xyz -> bool mask` applied per path to drop vertices
        (e.g. a cutout filter); a path left with < 2 vertices is skipped and its
        index is not consumed.

        Returns `(xyz, indices, next_index)`. Thread `next_index` into the next
        call to keep indices globally unique across several contour families.
        """
        chunks_xyz, chunks_idx = [], []
        path_idx = start
        for level, segs in level_paths:
            for xy in segs:
                if len(xy) < 2:
                    continue
                if height == "surface":
                    z = self.height_at(xy[:, 1], xy[:, 0])
                elif height == "level":
                    z = np.full(len(xy), level)
                elif height == "magnitude":
                    z = self.mag_at(xy[:, 1], xy[:, 0])
                else:
                    raise ValueError(f"unknown height policy {height!r}")
                xyz = np.column_stack([xy, z])
                if keep is not None:
                    xyz = xyz[keep(xyz)]
                    if len(xyz) < 2:
                        continue
                chunks_xyz.append(xyz)
                chunks_idx.append(np.full(len(xyz), path_idx, dtype=np.int64))
                path_idx += 1
        if not chunks_xyz:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), path_idx
        return np.concatenate(chunks_xyz), np.concatenate(chunks_idx), path_idx

    def grid_mesh(self, step=1, *, keep=None):
        """Heightfield triangulation of the clamped surface, subsampled by
        `step`. Vertices are (imag, real, z). Returns (vertices, triangles).

        `keep(im, re) -> bool array` restricts the mesh to a region: it is
        called on the (n_re, n_im) meshgrid of the subsampled vertices, in the
        emitted `(imag, real)` order, and a cell survives only if all four of
        its corners do. This is how a non-rectangular footprint — zeta's
        staircase cutout — becomes an occluder: the notch is empty because no
        triangle covers it, rather than because something masked it afterwards.
        """
        g_im = self.imag[::step]
        g_re = self.real[::step]
        g_z = self.clamped[::step, ::step]
        mask = None
        if keep is not None:
            IM, RE = np.meshgrid(g_im, g_re)
            mask = keep(IM, RE)
        return surface_grid_mesh(g_im, g_re, g_z, keep=mask)
