from dataclasses import dataclass, field

import numpy as np


def sample_grid(x_min, x_max, y_min, y_max, nx, ny=None):
    if ny is None:
        ny = nx
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    return x, y


def complex_grid(real_min, real_max, imag_min, imag_max, n_real, n_imag=None):
    real, imag = sample_grid(real_min, real_max, imag_min, imag_max, n_real, n_imag)
    return real[:, None] + 1j * imag, real, imag


def gradient_zones(
    func,
    real_bounds,
    imag_bounds,
    *,
    probe_res=400,
    percentile=95,
    pad=0.05,
    min_size=10,
    merge_overlapping=True,
):
    """Auto-discover high-gradient regions in `|func|` by probing a coarse grid.

    Returns a list of (r_min, r_max, i_min, i_max) bounding rectangles around
    connected components where `|∇ log|func||` exceeds `percentile` of the grid.
    log scaling matters here: gamma's gradient varies over many orders of
    magnitude, and we care about *relative* steepness.
    """
    from scipy import ndimage

    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds
    real = np.linspace(r_min, r_max, probe_res)
    imag = np.linspace(i_min, i_max, probe_res)
    grid = real[:, None] + 1j * imag
    values = np.abs(func(grid))

    log_v = np.log10(values + 1e-12)
    gy, gx = np.gradient(log_v)
    grad_mag = np.hypot(gy, gx)

    threshold = np.percentile(grad_mag, percentile)
    high = grad_mag > threshold
    labels, n_components = ndimage.label(high)

    zones = []
    pad_r = pad * (r_max - r_min)
    pad_i = pad * (i_max - i_min)
    for label_id in range(1, n_components + 1):
        mask = labels == label_id
        if mask.sum() < min_size:
            continue
        ys, xs = np.where(mask)
        r_lo = max(real[ys.min()] - pad_r, r_min)
        r_hi = min(real[ys.max()] + pad_r, r_max)
        i_lo = max(imag[xs.min()] - pad_i, i_min)
        i_hi = min(imag[xs.max()] + pad_i, i_max)
        zones.append((r_lo, r_hi, i_lo, i_hi))

    if merge_overlapping:
        zones = _merge_overlapping_rects(zones)
    return zones


def _merge_overlapping_rects(rects):
    """Iteratively merge any pair of rectangles that overlap, until stable."""
    rects = [tuple(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(rects)
        for i, a in enumerate(rects):
            if used[i]:
                continue
            r_lo, r_hi, i_lo, i_hi = a
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                br_lo, br_hi, bi_lo, bi_hi = rects[j]
                if (r_lo <= br_hi and br_lo <= r_hi and i_lo <= bi_hi and bi_lo <= i_hi):
                    r_lo, r_hi = min(r_lo, br_lo), max(r_hi, br_hi)
                    i_lo, i_hi = min(i_lo, bi_lo), max(i_hi, bi_hi)
                    used[j] = True
                    changed = True
            used[i] = True
            out.append((r_lo, r_hi, i_lo, i_hi))
        rects = out
    return rects


@dataclass
class AdaptiveSamples:
    """Pre-computed function values at coarse + per-zone fine resolutions.

    `coarse_values` is `func` evaluated on a (coarse_res, coarse_res) grid spanning
    `real_bounds × imag_bounds`. Each entry of `fine_values` is `func` evaluated on
    a sub-grid at proportional `fine_res` density inside the matching `fine_zones`
    rectangle.
    """
    coarse_values: np.ndarray
    real_bounds: tuple
    imag_bounds: tuple
    coarse_res: int
    fine_values: list = field(default_factory=list)
    fine_zones: list = field(default_factory=list)
    fine_resolutions: list = field(default_factory=list)


def sample_adaptive(
    func,
    real_bounds,
    imag_bounds,
    *,
    coarse_res,
    fine_res,
    fine_zones,
):
    """Evaluate `func` once at coarse_res over the full domain, then once per
    fine zone at proportional fine_res density.

    The fine grid spacing matches a hypothetical uniform fine_res grid: a zone
    spanning 20% of the real range gets `0.2 * fine_res` real samples.
    """
    r_min, r_max = real_bounds
    i_min, i_max = imag_bounds

    real = np.linspace(r_min, r_max, coarse_res)
    imag = np.linspace(i_min, i_max, coarse_res)
    coarse_values = func(real[:, None] + 1j * imag)

    fine_values = []
    fine_resolutions = []
    for (zr_min, zr_max, zi_min, zi_max) in fine_zones:
        n_real = max(4, int(round((zr_max - zr_min) / (r_max - r_min) * fine_res)))
        n_imag = max(4, int(round((zi_max - zi_min) / (i_max - i_min) * fine_res)))
        zreal = np.linspace(zr_min, zr_max, n_real)
        zimag = np.linspace(zi_min, zi_max, n_imag)
        zvalues = func(zreal[:, None] + 1j * zimag)
        fine_values.append(zvalues)
        fine_resolutions.append((n_real, n_imag))

    return AdaptiveSamples(
        coarse_values=coarse_values,
        real_bounds=real_bounds,
        imag_bounds=imag_bounds,
        coarse_res=coarse_res,
        fine_values=fine_values,
        fine_zones=list(fine_zones),
        fine_resolutions=fine_resolutions,
    )
