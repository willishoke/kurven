"""Before-and-after windows on two bakes of the same plate.

Each figure is one window of the plate, drawn twice from two SVG bakes: the
state before the branch on the left, the branch on the right. Each panel is
titled by the test or the construction that produced it, since the two
commits on the branch do different things. The ink is read straight out of
the SVGs the bake writes, so the figures show what the plate shows, with
every layer told apart by colour and counted.

    .venv/bin/python docs/wall-ink/figures.py            # draw every figure
    .venv/bin/python docs/wall-ink/figures.py back-wall  # just one

The bake writes its layers as anonymous <g> groups in the order the landscape
declares them (KurvenLandscape/Landscape.swift, `layers`), plus the fold-line
layer last when the bundle carries one. LAYERS below is that order; a bake
with a different layer set needs its own list.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent

# The two bakes: before the branch, and the branch.
BEFORE = HERE / "rgamma-pr8.svg"     # 5e5ec61
AFTER = HERE / "rgamma-walls.svg"    # wall-ink @ e138a3a

# Title, x range, y range in plate units, then the two panel titles, before
# and after. One figure per entry.
FIGURES = {
    "back-wall": (
        "The back wall, which faces away", (3.4, 5.3), (3.2, 5.6),
        "depth test: 5 hatch strokes leak through the margin",
        "n·e < 0: the hatch is dropped"),
    "right-wall": (
        "The right wall, which faces the eye", (4.2, 5.3), (0.5, 3.4),
        "depth test: the foot loses to its own face, 14 pieces",
        "n·e > 0 and the march is clear: the foot is one line"),
    "notch-5": (
        "The pit notch at -5", (-5.2, -4.8), (3.9, 4.3),
        "crest ends at the last sample under the cap",
        "crest gains the cap crossing, solved between samples"),
    "notch-4": (
        "The pit notch at -4", (-4.4, -3.6), (3.9, 4.4),
        "crest ends at the last sample under the cap",
        "crest gains the cap crossing, solved between samples"),
}

# Group order in the SVG, with a colour per role and a width per layer.
LAYERS = [
    ("mag_major",    "#1f3fd1", 1.1),
    ("ang_major",    "#d1281f", 1.1),
    ("mag_minor",    "#1f3fd1", 0.5),
    ("ang_minor",    "#d1281f", 0.5),
    ("cap_outline",  "#2f7a3e", 1.4),
    ("cap_hatch",    "#2f7a3e", 0.5),
    ("wall_outline", "#b8860b", 1.4),
    ("wall_hatch",   "#b8860b", 0.5),
    ("folds",        "#1c1a16", 1.4),
]

NS = {"svg": "http://www.w3.org/2000/svg"}


def read_layers(path):
    """Each layer's polylines, as lists of (x, y) in plate units.

    The bake writes plate coordinates with y up and flips them for the page
    with a single transform on the outer group, so the points are used as
    written."""
    outer = ET.parse(path).getroot().find("svg:g", NS)
    groups = outer.findall("svg:g", NS)
    if len(groups) != len(LAYERS):
        raise SystemExit(f"{path.name}: {len(groups)} layers, LAYERS names {len(LAYERS)}")
    layers = []
    for g in groups:
        paths = []
        for pl in g.findall("svg:polyline", NS):
            pts = [tuple(map(float, p.split(","))) for p in pl.get("points").split()]
            paths.append(pts)
        layers.append(paths)
    return layers


def in_window(path, xr, yr):
    return any(xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1] for x, y in path)


def draw(ax, layers, xr, yr, ncol):
    """Every path with a vertex in the window, and a legend counting them.

    Returns the number of legend rows, so the figure can leave room for them."""
    handles = []
    for (name, colour, width), paths in zip(LAYERS, layers):
        shown = [p for p in paths if in_window(p, xr, yr)]
        if not shown:
            continue
        for p in shown:
            xs, ys = zip(*p)
            # A one- or two-vertex fragment is invisible as a line at this
            # scale; a dot shows that the bake drew something there.
            marker = "o" if len(p) <= 2 else None
            ax.plot(xs, ys, color=colour, lw=width, marker=marker, ms=2.5,
                    solid_capstyle="round", solid_joinstyle="round")
        handles.append(Line2D([], [], color=colour, lw=max(width, 0.8),
                              label=f"{name} ({len(shown)})"))
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, length=2, colors="#6b655a")
    for side in ax.spines.values():
        side.set_color("#d9d4c8")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=ncol, fontsize=7, frameon=False, handlelength=1.6)
    return math.ceil(len(handles) / ncol)


def figure(name):
    title, xr, yr, before, after = FIGURES[name]
    w = xr[1] - xr[0]
    h = yr[1] - yr[0]
    scale = 5.2 / max(w, h)
    ncol = 4 if w >= h else 2
    # Room under the axes for the legend rows, over them for the two titles.
    legend_in = 0.5 + 0.19 * math.ceil(len(LAYERS) / ncol)
    title_in = 0.9
    # A tall window still needs a panel wide enough for its legend.
    fig_w = 2 * max(w * scale, 3.8) + 1.2
    fig_h = h * scale + legend_in + title_in
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
    for ax, label, path in zip(axes, (before, after), (BEFORE, AFTER)):
        draw(ax, read_layers(path), xr, yr, ncol)
        ax.set_title(label, fontsize=9, color="#1c1a16")
    fig.suptitle(f"{title}: x [{xr[0]}, {xr[1]}], y [{yr[0]}, {yr[1]}]",
                 fontsize=11, color="#1c1a16")
    fig.subplots_adjust(left=0.6 / fig_w, right=1 - 0.3 / fig_w,
                        bottom=legend_in / fig_h, top=1 - title_in / fig_h,
                        wspace=0.08)
    out = HERE / f"{name}.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(out.relative_to(HERE.parent.parent))


if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        figure(name)
