"""A landscape nobody wrote down in advance: any f, any window, any truncation.

The four plates in `examples/` are each a bespoke program. They know their own
function, the rectangle it is interesting on, where its poles are and how far up
each one should be cut, which edges are hatched and at what density, and which
contour levels read well -- all of it typed in, because all of it was decided by
a person looking at the picture. That is the right way to reproduce a plate from
1909 and the wrong way to let someone pick a function from a menu.

This module is the general form. A `Spec` is the whole of what a landscape is
before anyone looks at it: an expression, a rectangle, a resolution and a
truncation. Everything else -- which levels, which edges, how dense the hatching
-- is *derived* from those, by rules stated once here, and the derivation is
written into the manifest as a description rather than as the ink it produces.
Two things follow, and they are the point of the whole design:

  - the consumer can change its mind. A described layer is a question, so the
    cap, the spacing and the levels are all editable in the frontend without
    asking Python for anything, because the answer is recomputed from the grids
    that are already there (`kurven.hatch` is the rule, and the Swift reader
    implements the same one).
  - Python is left doing only the thing that actually needs it. Sampling f is
    the one stage that needs scipy and cannot be derived from a bundle; it is
    also the only stage that has to run again when the domain moves.

`CATALOG` is the menu: a preset is a name, an expression in `kurven.expr`, and
the window and truncation that make that function read as a landscape. It is not
a list of what the language can do -- the expression is the general thing and the
catalog is a set of good starting points for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from kurven.bundle import (AXES, Affine2, CameraPreset, Caps, Domain, Interval,
                           KeepAll, KeepBand, KeepBelowCap, KeepEvery,
                           KeepRegion, LayerCapHatch, LayerCapOutline,
                           LayerContour, LayerSpec, LayerWallHatch,
                           LayerWallOutline, NoCaps, PlateProjection, RealBand,
                           RealBandCaps, UniformCap, swap_from_world)
from kurven.expr import ExpressionError, canonical, compile_expression
from kurven.perimeter import Edge as LibEdge, Perimeter as LibPerimeter
from kurven.scene import InkLayer, Scene
from kurven.surface import Surface

#: Samples along the longer side of the domain. The other side gets whatever
#: keeps the cells square, so a wide, shallow window is not sampled as finely
#: across as along.
DEFAULT_RESOLUTION = 600

#: The occluder's heightfield is decimated to about this many samples a side.
#: The depth pass is vertex-bound, and a landscape being dragged is redrawn
#: every frame.
OCCLUDER_RESOLUTION = 800

#: Where a non-finite sample is put instead. A pole evaluates to inf, and inf in
#: a float32 heightfield is a texture full of nothing a rasterizer can use;
#: 1e12 is past any cap anyone will set and still a number.
HUGE = 1e12

#: How the plate looks by default: the reciprocal-factorial plate's camera,
#: which is the one of the four that frames an ordinary rectangular landscape.
PLATE = PlateProjection(0.5, -55.0, -90.0, True, None)
PLATE_MARGIN = 0.02
PLATE_BUFFER = 4000

#: Phase contours, as in every plate here: every 90 degrees major, every 30
#: minor. Unlike the magnitude levels these do not depend on the function -- arg
#: has the same range whatever f is.
PHASE_MAJOR = tuple(np.linspace(-np.pi, np.pi, 5))
PHASE_MINOR = tuple(v for v in np.linspace(-np.pi, np.pi, 13)
                    if not any(abs(v - m) < 1e-12 for m in PHASE_MAJOR))


# --------------------------------------------------------------------------
# the menu
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """A function worth looking at, and the window it is worth looking at in.

    `window` is how far the domain sliders reach, not what they start at: the
    interesting rectangle is `domain`, and the window is the room around it
    where the picture still means something. Both are here rather than computed
    because "how far out is this function still interesting" is a fact about the
    function -- ζ wants sixty units of imag and tan wants three.
    """

    name: str
    label: str
    expression: str
    domain: Domain
    window: Domain
    cap: float | None
    notes: str = ""

    def to_dict(self):
        return {"name": self.name, "label": self.label,
                "expression": self.expression,
                "domain": self.domain.to_dict(), "window": self.window.to_dict(),
                "cap": None if self.cap is None else float(self.cap),
                "notes": self.notes}


def _domain(r0, r1, i0, i1):
    return Domain(Interval(r0, r1), Interval(i0, i1))


CATALOG = (
    Preset("rgamma", "1/Γ(z) — reciprocal factorial", "1/gamma(z)",
           _domain(-5.5, 4.0, 0.0, 2.5), _domain(-12.0, 8.0, -6.0, 6.0), 5.0,
           "Entire, with zeros at the non-positive integers: the ripple along "
           "the negative real axis. Jahnke-Emde's Fig. 6."),
    Preset("gamma", "Γ(z) — gamma", "gamma(z)",
           _domain(-4.5, 4.5, 0.0, 2.5), _domain(-10.0, 10.0, -6.0, 6.0), 5.0,
           "A pole spire at each non-positive integer. Truncating per band "
           "cuts each spire at its own height, as the published plate does."),
    Preset("zeta", "ζ(z) — Riemann zeta", "zeta(z)",
           _domain(-6.0, 8.0, 0.0, 30.0), _domain(-20.0, 20.0, -40.0, 40.0), 6.0,
           "The pole at s = 1 and the zeros on the critical line. Sampled "
           "here rather than loaded from a cache."),
    Preset("cn", "cn(z, 0.64) — Jacobi elliptic", "cn(z, 0.64)",
           _domain(-4.0, 4.0, -3.5, 3.5), _domain(-12.0, 12.0, -12.0, 12.0), 4.0,
           "Doubly periodic: a lattice of poles, one per quarter-period cell."),
    Preset("tan", "tan z", "tan(z)",
           _domain(-4.8, 4.8, -1.6, 1.6), _domain(-12.0, 12.0, -4.0, 4.0), 5.0,
           "A pole at every half-period along the real axis, and flat plateaus "
           "of |tan| -> 1 away from it."),
    Preset("sin", "sin z", "sin(z)",
           _domain(-6.5, 6.5, -2.0, 2.0), _domain(-12.0, 12.0, -4.0, 4.0), 4.0,
           "Entire: zeros on the real axis, growing exponentially away from it."),
    Preset("exp_inv", "exp(1/z) — essential singularity", "exp(1/z)",
           _domain(-1.2, 1.2, -1.2, 1.2), _domain(-4.0, 4.0, -4.0, 4.0), 5.0,
           "Every value, infinitely often, in every neighbourhood of the "
           "origin. The landscape shows why."),
    Preset("cubic", "z³ - 1", "z^3 - 1",
           _domain(-1.8, 1.8, -1.8, 1.8), _domain(-5.0, 5.0, -5.0, 5.0), None,
           "Three zeros at the cube roots of unity; no poles, so no truncation."),
    Preset("moebius", "(z² - 1)/(z² + 1)", "(z^2 - 1)/(z^2 + 1)",
           _domain(-2.5, 2.5, -2.0, 2.0), _domain(-6.0, 6.0, -6.0, 6.0), 4.0,
           "Zeros at ±1, poles at ±i: two spires and two pits."),
    Preset("sqrt", "√z — a branch cut", "sqrt(z)",
           _domain(-2.5, 2.5, -2.0, 2.0), _domain(-6.0, 6.0, -6.0, 6.0), None,
           "The principal branch: the phase contours end on the cut along the "
           "negative real axis, where the surface is continuous and arg is not."),
    Preset("log", "log z", "log(z)",
           _domain(-2.5, 2.5, -2.0, 2.0), _domain(-6.0, 6.0, -6.0, 6.0), 3.0,
           "A logarithmic pole at the origin: the one spire that grows slowly "
           "enough to see the shape of."),
    Preset("erf", "erf z", "erf(z)",
           _domain(-3.0, 3.0, -2.5, 2.5), _domain(-8.0, 8.0, -8.0, 8.0), 5.0,
           "Entire, but it grows like exp(z²) off the real axis."),
    Preset("besselj", "J₀(z)", "besselj(0, z)",
           _domain(-10.0, 10.0, -3.0, 3.0), _domain(-25.0, 25.0, -8.0, 8.0), 4.0,
           "Oscillating and decaying along the real axis, growing off it."),
    Preset("digamma", "ψ(z) — digamma", "digamma(z)",
           _domain(-4.5, 4.5, 0.0, 2.5), _domain(-10.0, 10.0, -6.0, 6.0), 5.0,
           "Γ's logarithmic derivative: simple poles at the non-positive "
           "integers, and no zeros in sight."),
)


def preset(name):
    for p in CATALOG:
        if p.name == name:
            return p
    raise KeyError(f"no preset {name!r}; have {[p.name for p in CATALOG]}")


# --------------------------------------------------------------------------
# the derived numbers
# --------------------------------------------------------------------------


def nice(x, *, down=False):
    """`x` rounded to a number a person would have chosen: 1, 2, 2.5 or 5 times
    a power of ten. Up by default; `down` for a spacing, where rounding up
    would thin the hatching rather than thicken it."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    scale = 10.0 ** np.floor(np.log10(x))
    m = x / scale
    steps = (1.0, 2.0, 2.5, 5.0, 10.0)
    if down:
        return float(scale * max(s for s in steps if s <= m * (1 + 1e-12)))
    return float(scale * min(s for s in steps if s >= m * (1 - 1e-12)))


def magnitude_levels(ceiling):
    """Major and minor |f| levels under a ceiling.

    Majors about ten of them, minors five to a major -- the ratio the plates
    use. The levels are multiples of the step rather than a spread between two
    endpoints, so raising the cap adds contours at the top instead of moving
    every contour already drawn.
    """
    ceiling = float(ceiling)
    if not np.isfinite(ceiling) or ceiling <= 0:
        return (), ()
    step = nice(ceiling / 10.0)
    minor_step = step / 5.0
    major = tuple(round(step * k, 10) for k in range(1, int(ceiling / step) + 1))
    minor = tuple(round(minor_step * k, 10)
                  for k in range(1, int(ceiling / minor_step) + 1)
                  if not any(abs(minor_step * k - m) < 1e-9 for m in major))
    return major, minor


def default_caps(mag):
    """How far up to cut, read off the samples.

    A pole makes the top of the landscape unboundedly tall, and a picture of it
    is a needle and a floor. The rule: if the highest sample is far above the
    bulk of them -- more than three times the 99th percentile -- there is a
    spire, and it is cut at a round number near that percentile. Otherwise the
    function is bounded on this window and truncating it would invent a
    plateau that is not there.
    """
    finite = mag[np.isfinite(mag)]
    if finite.size == 0:
        return NoCaps()
    q = float(np.quantile(finite, 0.99))
    top = float(finite.max())
    if q <= 0 or top <= 3 * q:
        return NoCaps()
    return UniformCap(nice(q))


def ceiling_of(caps, mag):
    """The highest |f| worth drawing a contour at: the cap when there is one,
    else a robust maximum of the samples (the very top of an uncapped spire is
    one sample tall and contours there are noise)."""
    if isinstance(caps, UniformCap):
        return float(caps.z)
    if isinstance(caps, RealBandCaps):
        tops = [b.cap for b in caps.bands] + ([caps.beyond]
                                              if np.isfinite(caps.beyond) else [])
        return float(max(tops)) if tops else 1.0
    finite = mag[np.isfinite(mag)]
    return float(np.quantile(finite, 0.999)) if finite.size else 1.0


def grid_shape(domain, resolution):
    """`(n_real, n_imag)` for a resolution given along the longer side, with
    square cells."""
    dr = abs(domain.real.hi - domain.real.lo)
    di = abs(domain.imag.hi - domain.imag.lo)
    longest = max(dr, di)
    if longest <= 0:
        return max(int(resolution), 2), max(int(resolution), 2)
    n_real = max(int(round(resolution * dr / longest)), 8)
    n_imag = max(int(round(resolution * di / longest)), 8)
    return n_real, n_imag


def cell_size(domain, shape):
    n_real, n_imag = shape
    return (abs(domain.real.hi - domain.real.lo) / max(n_real - 1, 1),
            abs(domain.imag.hi - domain.imag.lo) / max(n_imag - 1, 1))


def hatch_spacing(domain):
    """One number for both hatchings: about seventy strokes along the longest
    side, rounded down to something a ruler would have."""
    span = max(abs(domain.real.hi - domain.real.lo),
               abs(domain.imag.hi - domain.imag.lo))
    return nice(span / 70.0, down=True)


# --------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """Everything a landscape is before anyone looks at it.

    `caps` and `layers` are the *styling*, and both may be absent, meaning "the
    rules above decide". The frontend sends them back when it has edited them,
    so that moving the domain keeps the truncation and the hatching the user
    chose instead of resetting to the defaults for the new window.
    """

    expression: str
    domain: Domain
    resolution: int = DEFAULT_RESOLUTION
    caps: Caps | None = None
    #: World units between hatch strokes; absent means "read it off the window".
    #: Separate from `layers` because it is one number a slider moves, where
    #: `layers` is the whole styling and is only sent back once it has been
    #: edited wholesale.
    spacing: float | None = None
    layers: tuple[LayerSpec, ...] | None = None
    #: The catalog entry this came from, when it came from one. Provenance
    #: only: the expression is what is evaluated.
    name: str = ""

    @classmethod
    def of(cls, name, **overrides):
        """The spec a catalog preset stands for."""
        p = preset(name)
        base = cls(expression=p.expression, domain=p.domain, name=p.name,
                   caps=None if p.cap is None else UniformCap(p.cap))
        return replace(base, **overrides) if overrides else base

    def to_dict(self):
        return {"expression": self.expression, "domain": self.domain.to_dict(),
                "resolution": int(self.resolution), "name": self.name,
                "spacing": None if self.spacing is None else float(self.spacing),
                "caps": None if self.caps is None else self.caps.to_dict(),
                "layers": None if self.layers is None
                          else [l.to_dict() for l in self.layers]}

    @classmethod
    def from_dict(cls, d):
        from kurven.bundle import _require

        caps = d.get("caps")
        layers = d.get("layers")
        return cls(
            expression=str(_require(d, "expression", "Spec")),
            domain=Domain.from_dict(_require(d, "domain", "Spec")),
            resolution=int(d.get("resolution", DEFAULT_RESOLUTION)),
            caps=None if caps is None else Caps.from_dict(caps),
            spacing=None if d.get("spacing") is None else float(d["spacing"]),
            layers=None if layers is None
                   else tuple(LayerSpec.from_dict(l) for l in layers),
            name=str(d.get("name", "")))


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def evaluator(expression):
    """The compiled expression, with every non-finite value pushed to `HUGE`.

    A pole is where the interesting part of a landscape is, and it is also
    where f is inf, nan, or a number so large that float32 cannot hold it. The
    surface is capped long before any of that matters, so the honest thing is a
    height the cap will cut rather than a hole in the grid: nan in a heightfield
    propagates into the depth buffer and takes the picture with it.
    """
    f = compile_expression(expression)

    def sampled(z):
        values = np.asarray(f(z), dtype=complex)
        bad = ~np.isfinite(values)
        if bad.any():
            values = np.where(bad, HUGE, values)
        big = np.abs(values) > HUGE
        if big.any():
            values = np.where(big, HUGE * np.exp(1j * np.angle(values)), values)
        return values

    return sampled


def sample(spec, *, verbose=False):
    """Evaluate the expression over the spec's rectangle. The expensive half."""
    n_real, n_imag = grid_shape(spec.domain, spec.resolution)
    real = np.linspace(spec.domain.real.lo, spec.domain.real.hi, n_real)
    imag = np.linspace(spec.domain.imag.lo, spec.domain.imag.hi, n_imag)
    f = evaluator(spec.expression)
    surface = Surface.from_function(f, real, imag)
    surface.caps = spec.caps if spec.caps is not None else default_caps(surface.mag)
    if verbose:
        finite = surface.mag[np.isfinite(surface.mag)]
        print(f"{canonical(spec.expression)}  {n_real}x{n_imag}  "
              f"|f| <= {finite.max() if finite.size else 0:.3g}  cap {surface.caps}")
    return surface


# --------------------------------------------------------------------------
# the default styling
# --------------------------------------------------------------------------


def rectangle(domain):
    """The domain's boundary as a closed traversal in `(imag, real)`.

    `Perimeter.rectangle` emits front/back/left/right, which is a set of walls
    rather than a loop, and a loop is what puts one post at each corner and
    lets the region answer `contains`. The edge order here is the traversal
    front, right, back, left -- so edge 0 is the real axis at the near side,
    the one a plate hatches when it hatches only one.
    """
    r0, r1 = domain.real.lo, domain.real.hi
    i0, i1 = domain.imag.lo, domain.imag.hi
    return LibPerimeter([
        LibEdge((i0, r0), (i0, r1)),      # front: imag low, along real
        LibEdge((i0, r1), (i1, r1)),      # right: real high, along imag
        LibEdge((i1, r1), (i1, r0)),      # back
        LibEdge((i1, r0), (i0, r0)),      # left
    ])


def wall_density(domain, shape):
    """Curtain samples per edge: one per grid cell along that edge, so a wall
    follows the surface as closely as the surface is known."""
    dx, dy = cell_size(domain, shape)
    along_real = max(int(round(abs(domain.real.hi - domain.real.lo) / max(dx, 1e-12))) + 1, 2)
    along_imag = max(int(round(abs(domain.imag.hi - domain.imag.lo) / max(dy, 1e-12))) + 1, 2)
    return [min(along_real, 4096), min(along_imag, 4096),
            min(along_real, 4096), min(along_imag, 4096)]


def default_layers(domain, shape, caps, ceiling, *, spacing=None, phase=True):
    """The whole plate, as descriptions.

    Draw order is declaration order: the two major families, the two minor
    ones, then the scaffold that frames them. The magnitude families are cut at
    the cap (`KeepBelowCap`) because a contour at a level above the truncation
    would otherwise float in the air over the plateau that replaced its spire;
    the phase families are lifted to the *unclamped* magnitude and cut the same
    way, which is what leaves a truncated top blank except for its hatching.
    """
    spacing = hatch_spacing(domain) if spacing is None else float(spacing)
    dx, dy = cell_size(domain, shape)
    pitch = max(dx, dy)
    major, minor = magnitude_levels(ceiling)
    edges = (0, 1, 2, 3)

    layers = [
        LayerSpec("mag_major", "magnitude",
                  LayerContour("magnitude", major, KeepBelowCap()), 0.4, "level"),
    ]
    if phase:
        layers.append(
            LayerSpec("ang_major", "phase",
                      LayerContour("phase", PHASE_MAJOR, KeepAll()),
                      0.4, "surface"))
    layers.append(
        LayerSpec("mag_minor", "magnitude",
                  LayerContour("magnitude", minor, KeepBelowCap()), 0.15, "level"))
    if phase:
        layers.append(
            LayerSpec("ang_minor", "phase",
                      LayerContour("phase", PHASE_MINOR, KeepAll()),
                      0.15, "surface"))
    layers += [
        LayerSpec("cap_outline", "scaffold", LayerCapOutline(), 0.4, "level"),
        LayerSpec("cap_hatch", "scaffold", LayerCapHatch("real", spacing),
                  0.2, "level"),
        LayerSpec("wall_outline", "scaffold", LayerWallOutline(edges, pitch),
                  0.3, "surface"),
        LayerSpec("wall_hatch", "scaffold",
                  LayerWallHatch(edges, spacing, pitch), 0.25, "surface"),
    ]
    return tuple(layers)


# --------------------------------------------------------------------------
# geometry, for the plate
# --------------------------------------------------------------------------


def keep_predicate(keep, surface, world_perimeter):
    """A `Keep` as the `lift_contours` filter it means.

    Library order in, since that is what `lift_contours` passes: column 0 is
    imag, column 1 is real. `Surface.admits` on the Swift side is the same
    predicate in world order, down to the strict inequalities.
    """
    if isinstance(keep, KeepAll):
        return None
    if isinstance(keep, KeepBelowCap):
        caps = surface.caps or NoCaps()
        return lambda xyz: xyz[:, 2] <= caps.at(xyz[:, 1])
    if isinstance(keep, KeepRegion):
        if world_perimeter is None:
            return None
        return lambda xyz: world_perimeter.contains(xyz[:, 1], xyz[:, 0])
    if isinstance(keep, KeepBand):
        col = 1 if keep.axis == "real" else 0
        return lambda xyz: (xyz[:, col] > keep.lo) & (xyz[:, col] < keep.hi)
    if isinstance(keep, KeepEvery):
        parts = [keep_predicate(k, surface, world_perimeter) for k in keep.of]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return lambda xyz: np.logical_and.reduce([p(xyz) for p in parts])
    raise ValueError(f"unknown keep {keep!r}")


def contour_geometry(spec_layer, surface, world_perimeter, *, start, chunk_count):
    """One described contour layer, contoured and lifted."""
    from kurven.contours import contour_levels

    source = spec_layer.source
    grid = surface.mag if source.field == "magnitude" else surface.angle
    rb = (float(surface.real[0]), float(surface.real[-1]))
    ib = (float(surface.imag[0]), float(surface.imag[-1]))
    levels = contour_levels(grid, list(source.levels), rb, ib,
                            chunk_count=chunk_count)
    return surface.lift_contours(
        levels, start=start, height=spec_layer.height_policy,
        keep=keep_predicate(source.keep, surface, world_perimeter))


def hatch_geometry(spec_layer, surface, world_perimeter, region, *, chunk_count):
    """One described hatching layer, as library-order segments.

    The heights come from the analytic function here -- this is the Python
    plate, which has f -- where the consumer's come from the grid. That
    difference is the grid's interpolation error and is the thing
    `tests/compare_bake.py --derived` measures.
    """
    from kurven import hatch

    domain = Domain(Interval(float(surface.real[0]), float(surface.real[-1])),
                    Interval(float(surface.imag[0]), float(surface.imag[-1])))
    grid = surface.mag.T
    paths = hatch.derive(
        spec_layer.source,
        perimeter=world_perimeter, region=region, tiles=(Affine2.identity(),),
        height=lambda x, y: surface.height_at(x, y),
        # The cap strokes read the *grid*, even here, where f is available: a
        # stroke ends where |f| crosses the cap between two samples, and that
        # crossing is a statement about the samples. Solved analytically it
        # would end somewhere the plateau it lies on -- which is meshed from
        # the grid -- does not.
        magnitude=hatch.grid_sampler(grid, domain),
        height_grid=grid, domain=domain, caps=surface.caps or NoCaps(),
        chunk_count=chunk_count)
    if paths is None:
        return None
    return [swap_from_world(p) for p in paths]


# --------------------------------------------------------------------------
# the scene
# --------------------------------------------------------------------------


def build_scene(spec, *, verbose=True, geometry=True, chunk_count=1):
    """A `Scene` for the spec: the camera-independent half of a landscape.

    With `geometry=False` the layers are their descriptions and nothing else --
    no contouring, no hatching. That is what the service wants: a bundle written
    with `--derived` carries no vertices, so computing them to throw them away
    would be the slowest part of answering a request to move a slider.
    """
    surface = sample(spec, verbose=verbose)
    shape = (len(surface.real), len(surface.imag))
    domain = Domain(Interval(float(surface.real[0]), float(surface.real[-1])),
                    Interval(float(surface.imag[0]), float(surface.imag[-1])))
    caps = surface.caps
    perim = rectangle(domain)
    density = wall_density(domain, shape)
    world_perimeter = perim.to_world(density)

    specs = spec.layers
    if specs is None:
        specs = default_layers(domain, shape, caps,
                               ceiling_of(caps, surface.mag),
                               spacing=spec.spacing, phase=True)

    layers = []
    index = 0
    for spec_layer in specs:
        if not geometry:
            layers.append(InkLayer(spec_layer.name, spec_layer.role,
                                   np.zeros((0, 3)), np.zeros(0, dtype=np.int64),
                                   spec_layer.width, spec_layer.height_policy,
                                   spec_layer.color, spec_layer.clipped,
                                   source=spec_layer.source))
            continue
        segments = hatch_geometry(spec_layer, surface, world_perimeter, None,
                                  chunk_count=chunk_count)
        if segments is not None:
            layers.append(InkLayer.from_segments(
                spec_layer.name, spec_layer.role, segments, spec_layer.width,
                height_policy=spec_layer.height_policy, color=spec_layer.color,
                clipped=spec_layer.clipped, source=spec_layer.source))
            continue
        if isinstance(spec_layer.source, LayerContour):
            xyz, indices, index = contour_geometry(
                spec_layer, surface, world_perimeter, start=index,
                chunk_count=chunk_count)
            layers.append(InkLayer(spec_layer.name, spec_layer.role, xyz, indices,
                                   spec_layer.width, spec_layer.height_policy,
                                   spec_layer.color, spec_layer.clipped,
                                   source=spec_layer.source))
            continue
        raise ValueError(f"layer {spec_layer.name!r}: a landscape cannot build "
                         f"{spec_layer.source!r} -- it carries no dumped ink")

    if verbose and geometry:
        for l in layers:
            print(f"  {l.name:<13} {len(l.runs()):>6} paths")

    return Scene(
        function=canonical(spec.expression),
        params={"expression": canonical(spec.expression), "name": spec.name,
                "resolution": int(spec.resolution),
                "rMin": float(domain.real.lo), "rMax": float(domain.real.hi),
                "iMin": float(domain.imag.lo), "iMax": float(domain.imag.hi),
                "nReal": shape[0], "nImag": shape[1]},
        surface=surface,
        layers=tuple(layers),
        preset=CameraPreset("plate", PLATE, PLATE_MARGIN, PLATE_BUFFER),
        caps=caps,
        occluder_step=max(1, max(shape) // OCCLUDER_RESOLUTION),
        tiles=(Affine2.identity(),),
        perimeter=world_perimeter,
        walls=tuple(perim.wall_curtains(surface, density)),
    )
