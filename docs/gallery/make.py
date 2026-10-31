"""The README's figures, baked from the formula by the Swift CLI.

    swift build -c release
    python3 docs/gallery/make.py

Each figure is a native landscape (`kurven-cli landscape`) given a camera and,
for gamma, the published plate's truncation, then baked (`kurven-cli bake`,
which places the contours on f and tests every curve's visibility exactly). Standard library
only: nothing here computes a line, it only sets up the bundles and lays the
results out. A white ground is added so the plates read in a dark theme.

The bake keeps every vertex the refiner placed, about two to the pixel at
README width. Safari strokes chains that fine with a visible wobble, so each
polyline is thinned here to the vertices it needs to stay within
`SIMPLIFY_PX` of itself, which leaves segments a few pixels long.
"""
import json, math, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / ".build/release/kurven-cli"
OUT = Path(__file__).resolve().parent
PAPER = "#ffffff"
# The bake's widths are for a sixteen-inch page; a README column is 880 px.
# Scale every stroke alike so the heaviest is a fine line there, keeping the
# plate's hierarchy of major over minor lines.
README_PX = 880
HEAVIEST_PX = 0.5
# The bake's viewBox is tight to the strokes; a plate sits on its page with this
# much paper on every side, as a fraction of the drawing's width.
MARGIN = 0.1
# The landscape hatches the plateaus and walls at about seventy lines along the
# window's longer side; the plates hatch them this much closer.
HATCH = 0.75
# How far, in README pixels, a thinned polyline may stray from the bake's.
SIMPLIFY_PX = 0.05


def run(*args):
    r = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"kurven-cli {' '.join(map(str, args))}\n{r.stderr}")
    return r.stdout


def landscape(work, name, expression, re_, im, res, truncation, cameras):
    """Sample `expression` and give the bundle one preset per camera.

    `truncation` is the truncation height, or `(bands, beyond)` for a height per band
    of Re z: `bands` as (below, height) pairs. It is set when the function is
    sampled, because the contour levels are chosen then, up to the highest."""
    bundle = work / f"{name}.kurven"
    if isinstance(truncation, tuple):
        bands, beyond = truncation
        flags = ["--bands", ",".join(f"{b}:{h}" for b, h in bands), "--beyond", beyond]
    else:
        flags = ["--truncation", truncation]
    run("landscape", expression, "--re", f"{re_[0]},{re_[1]}", "--im", f"{im[0]},{im[1]}",
        "--res", res, *flags, "-o", bundle)
    path = bundle / "manifest.json"
    m = json.loads(path.read_text())
    base = m["presets"][0]
    m["presets"] = [dict(base, name=key, projection=dict(base["projection"], **projection))
                    for key, projection in cameras.items()]
    for layer in m["layers"]:
        source = layer["source"]
        if source["kind"] == "hatching" and source["faces"]["kind"] in ("plateaus", "walls"):
            source["spacing"] *= HATCH
    path.write_text(json.dumps(m, indent=1))
    return bundle


def bake(bundle, preset, svg):
    run("bake", bundle, "--preset", preset, "-o", svg)
    return svg.read_text()


def thinned(points, tol):
    """`points` reduced by Douglas–Peucker to those within `tol` of the line."""
    keep, stack = {0, len(points) - 1}, [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        (ax, ay), (bx, by) = points[i], points[j]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        far, k = -1.0, None
        for n in range(i + 1, j):
            px, py = points[n][0] - ax, points[n][1] - ay
            d = abs(px * dy - py * dx) / length if length > 1e-12 else math.hypot(px, py)
            if d > far:
                far, k = d, n
        if far > tol:
            keep.add(k)
            stack += [(i, k), (k, j)]
    return [points[n] for n in sorted(keep)]


def simplified(svg, px_across):
    """`svg` with every polyline thinned to within `SIMPLIFY_PX` when its
    viewBox is shown `px_across` pixels wide."""
    across = float(re.search(r'viewBox="([^"]+)"', svg).group(1).split()[2])
    tol = SIMPLIFY_PX * across / px_across

    def thin(m):
        points = [tuple(map(float, p.split(","))) for p in m.group(1).split()]
        return 'points="' + " ".join(f"{x:.5f},{y:.5f}" for x, y in thinned(points, tol)) + '"'
    return re.sub(r'points="([^"]+)"', thin, svg)


def scaled(svg, px_across):
    """`svg` with every stroke scaled so the heaviest is `HEAVIEST_PX` when its
    viewBox is shown `px_across` pixels wide."""
    across = float(re.search(r'viewBox="([^"]+)"', svg).group(1).split()[2])
    widths = [float(w) for w in re.findall(r'stroke-width="([0-9.e-]+)"', svg)]
    k = HEAVIEST_PX * across / (px_across * max(widths))
    return re.sub(r'stroke-width="([0-9.e-]+)"', lambda m: f'stroke-width="{float(m.group(1)) * k:.6g}"', svg)


def on_paper(svg):
    """The bake's SVG with `MARGIN` of paper round the drawing and a ground
    under it."""
    x, y, w, h = map(float, re.search(r'viewBox="([^"]+)"', svg).group(1).split())
    pad = MARGIN * w
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    px = float(re.search(r'<svg[^>]*\swidth="([0-9.]+)"', svg).group(1))
    svg = re.sub(r'viewBox="[^"]+"', f'viewBox="{x:.4f} {y:.4f} {w:.4f} {h:.4f}"', svg, count=1)
    svg = re.sub(r'(<svg[^>]*\s)height="[0-9.]+"', rf'\g<1>height="{px * h / w:.3f}"', svg, count=1)
    rect = f'<rect x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" fill="{PAPER}"/>'
    return re.sub(r"(<svg[^>]*>)", r"\1" + rect, svg, count=1)


def grid(svgs, cols, cell=1000, gap=2 * MARGIN):
    """Tiles at one scale: each bake's drawing centred in a square cell whose
    side is the largest extent among them."""
    parts = []
    for s in svgs:
        x, y, w, h = map(float, re.search(r'viewBox="([^"]+)"', s).group(1).split())
        body = re.search(r"<svg[^>]*>(.*)</svg>", s, re.S).group(1)
        parts.append(((x + w / 2, y + h / 2), max(w, h), body))
    side = max(p[1] for p in parts) * (1 + gap)
    tile = f'<svg viewBox="0 0 {side} {side}">'
    parts = [(c, e, simplified(scaled(tile + b + "</svg>", README_PX / cols), README_PX / cols)[len(tile):-len("</svg>")])
             for c, e, b in parts]
    rows = math.ceil(len(parts) / cols)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cols * cell} {rows * cell}" '
           f'width="{cols * cell}" height="{rows * cell}">',
           f'<rect width="{cols * cell}" height="{rows * cell}" fill="{PAPER}"/>']
    for i, ((cx, cy), _, body) in enumerate(parts):
        r, c = divmod(i, cols)
        out.append(f'<svg x="{c * cell}" y="{r * cell}" width="{cell}" height="{cell}" '
                   f'viewBox="{cx - side / 2:.4f} {cy - side / 2:.4f} {side:.4f} {side:.4f}">{body}</svg>')
    out.append("</svg>")
    return "\n".join(out)


def agm(a, b):
    while abs(a - b) > 1e-15 * a:
        a, b = (a + b) / 2, math.sqrt(a * b)
    return a


def main():
    if not CLI.exists():
        sys.exit(f"build the CLI first: swift build -c release")
    work = Path(tempfile.mkdtemp(prefix="kurven-gallery-"))
    try:
        # Γ, as the published plate frames it: the strip |Im z| ≤ 2.5, a truncation
        # height per pole band, and that plate's camera.
        gamma = landscape(work, "gamma", "gamma(z)", (-4.5, 4.5), (-2.5, 2.5), 1000,
                          ([(-3.5, 2.5), (-2.5, 3.0), (-1.5, 4.0), (0.5, 5.5)], 4.0),
                          {"plate": dict(shear=-0.5, xAngle=-55.0, zAngle=-90.0, flipX=False)})
        (OUT / "gamma.svg").write_text(simplified(scaled(on_paper(bake(gamma, "plate", work / "gamma.svg")), README_PX), README_PX))

        # cn(z, 0.64) over the elliptic plate's block of quarter periods,
        # [−K, 5K] × [0, 3K′], with that plate's camera.
        m = 0.64
        K = math.pi / (2 * agm(1, math.sqrt(1 - m)))
        Kp = math.pi / (2 * agm(1, math.sqrt(m)))
        cn = landscape(work, "cn", f"cn(z, {m})", (-K, 5 * K), (0, 3 * Kp), 1200, 4,
                       {"plate": dict(shear=0.51, xAngle=-63.0, zAngle=-90.0, flipX=True)})
        (OUT / "cn.svg").write_text(simplified(scaled(on_paper(bake(cn, "plate", work / "cn.svg")), README_PX), README_PX))

        # 1/(z³ − 1) from four oblique corners: the front face head-on from
        # either side (turn ±90°), the depth receding right or left (shear ±½).
        corners = {f"t{z}s{s}".replace("-", "m"): dict(shear=s, xAngle=-55.0, zAngle=float(z), flipX=True)
                   for z in (-90, 90) for s in (0.5, -0.5)}
        cubic = landscape(work, "cubic", "1/(z^3 - 1)", (-1.3, 1.6), (-1.2, 1.5), 300, 5, corners)
        tiles = [bake(cubic, key, work / f"cubic-{key}.svg") for key in corners]
        (OUT / "cubic-corners.svg").write_text(grid(tiles, 2))

        # The picture frame of docs/solids.md: its edges alone, then its ten
        # faces hatched with Turing's pattern in four widths.
        frame = ["surface", "solid", "--kind", "frame", "--size", 2, "--hole", 2, "--depth", 0.8,
                 "--x-angle", -55, "--z-angle", -60, "--shear", 0, "--width", 0.6]
        run(*frame, "-o", work / "frame-edges.svg")
        run(*frame, "--hatching", "turing", "--angle", 30, "--spacing", 0.035, "--width-steps", 4,
            "--scale", 0.18, "--seed", 3, "--hatch-width", 0.5, "-o", work / "frame-turing.svg")
        tiles = [(work / f"frame-{k}.svg").read_text() for k in ("edges", "turing")]
        (OUT / "frame.svg").write_text(grid(tiles, 2))
    finally:
        shutil.rmtree(work)
    for f in ("gamma.svg", "cn.svg", "cubic-corners.svg", "frame.svg"):
        print(OUT / f, f"{(OUT / f).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
