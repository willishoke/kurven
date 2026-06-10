# kurven

A Python library for rendering **analytic landscapes** — the 3D surface visualization of complex functions pioneered by Jahnke and Emde in their 1909 atlas *Tafeln höherer Funktionen*. The height of the surface at each point z is |f(z)|; magnitude and phase contour lines are projected onto the surface and clipped against a depth buffer to produce a hidden-line-removed vector graphic.

## Gallery

### Γ(z) — Gamma function

![Gamma function analytic landscape](docs/gamma_hi_res.svg)

The upper half-plane Re(z) ∈ [−4.5, 4.5], Im(z) ∈ [0, 2.5]. Four pole spires rise at the non-positive integers; the surface is truncated at a consistent height per spire so each cap is visible from above.

---

## Algorithm

The pipeline has six stages: **sample → contour → lift → project → depth-buffer → clip**.

### 1. Sample

Evaluate f(z) on a uniform grid of complex values. For functions with poles or rapid variation (e.g. Γ near the negative integers), an **adaptive sampler** (`kurven/sampling.py`) first probes a coarse grid to locate high-gradient regions via |∇ log|f||, then re-samples those zones at a finer density. This concentrates resolution where contours are densest without paying for it everywhere.

### 2. Contour

Extract iso-lines of |f(z)| and arg(f(z)) at a chosen set of levels using marching squares (via [contourpy](https://contourpy.readthedocs.io/)). Two families of curves are generated — magnitude and phase — each at major and minor spacings. The contourpy threaded backend parallelizes across levels; chunk-boundary seams are stitched by exact endpoint matching (`_stitch_chunk_seams`).

For functions with adaptive sampling, coarse contours that fall inside a fine zone are dropped; fine contours fill in. Where a coarse and fine path meet at a zone boundary they are welded together (`_stitch_paths`).

### 3. Lift to 3D

Each contour vertex (x, y) in the complex plane is lifted to (x, y, |f(x + iy)|). This turns the flat contour diagram into a network of curves draped over the magnitude surface.

### 4. Project

An isometric-style projection:

1. **Shear** — y′ = y + s·x (default s = 0.5) introduces foreshortening perpendicular to the viewer, matching the visual style of the original Jahnke-Emde plates.
2. **Rotate** — a Z rotation followed by an X rotation tilts the surface toward the viewer.

The result is a 3D point cloud in screen space.

### 5. Depth buffer

The magnitude surface is independently meshed as a triangulated grid and rasterized into a Z-buffer (`kurven/zbuffer.py`). Each pixel stores the maximum depth value seen so far. A GPU path via [moderngl](https://moderngl.readthedocs.io/) rasterizes with `GL_MAX` blending; the CPU fallback rasterizes triangle-by-triangle using barycentric interpolation.

### 6. Clip and outline

**Hidden-line removal** (`clip_hidden_lines`): each contour point is checked against the Z-buffer. Points behind the surface are invisible; the remaining runs are split into visible segments.

**Silhouette** (`extract_outline`): the Z-buffer is binarized (filled vs. empty) and the level-0 contour gives the outer boundary of the rendered shape — the "picture frame" silhouette.

Both are saved as SVG polylines via matplotlib.

---

## Project structure

```
kurven/
  sampling.py    — uniform + adaptive grid evaluation, gradient-zone discovery
  contours.py    — marching-squares extraction, seam stitching, path lifting
  projection.py  — isometric shear + rotation
  zbuffer.py     — Z-buffer class, CPU and GPU triangle rasterizers
  outline.py     — hidden-line clipping, silhouette extraction
  pipeline.py    — thin convenience wrapper for the generic stages

examples/
  gamma.py       — Γ(z): faithful reproduction of the Jahnke-Emde gamma plate
  elliptic.py    — cn(z, m): Jacobi elliptic function landscape
  zeta.py        — ζ(s): Riemann zeta function (work in progress)
```

## Installation

```bash
uv sync            # CPU only
uv sync --extra gpu  # + moderngl for GPU rasterization
```

## Running the examples

```bash
# Gamma (default: res=10000, buffer=20000 — takes ~10 min on CPU, ~1 min with GPU)
python examples/gamma.py --gpu

# Smoke-test at lower res
python examples/gamma.py --res 800 --buffer 1600

# Elliptic cn
python examples/elliptic.py --gpu

# Both write <prefix>_hi_res.svg (and gamma also writes <prefix>_raw.svg)
python examples/gamma.py --gpu --output-prefix out/my_gamma
```

## References

- Jahnke, E. & Emde, F. (1909). *Tafeln höherer Funktionen*. Teubner.
- Needham, T. (1997). *Visual Complex Analysis*. Oxford University Press.
