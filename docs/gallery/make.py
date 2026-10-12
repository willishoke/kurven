"""The README's figures, baked from the formula by the Swift CLI.

    swift build -c release --package-path KurvenSwift
    python3 docs/gallery/make.py

Each figure is a native landscape (`kurven-cli landscape`) given a camera and,
for gamma, the published plate's caps, then baked (`kurven-cli bake`, which
places the contours on f and judges every line exactly). Standard library
only: nothing here computes a line, it only sets up the bundles and lays the
results out. A white ground is added so the plates read in a dark theme.
"""
import json, math, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "KurvenSwift/.build/release/kurven-cli"
OUT = Path(__file__).resolve().parent
PAPER = "#ffffff"
# The bake's widths are for a sixteen-inch page; a README column is 880 px.
# Scale every stroke alike so the heaviest is about a pixel there, keeping
# the plate's hierarchy of major over minor lines.
README_PX = 880
HEAVIEST_PX = 1.1


def run(*args):
    r = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"kurven-cli {' '.join(map(str, args))}\n{r.stderr}")
    return r.stdout


def landscape(work, name, expression, re_, im, res, cap, cameras, caps=None):
    """Sample `expression` and give the bundle one preset per camera."""
    bundle = work / f"{name}.kurven"
    run("landscape", expression, "--re", f"{re_[0]},{re_[1]}", "--im", f"{im[0]},{im[1]}",
        "--res", res, "--cap", cap, "-o", bundle)
    path = bundle / "manifest.json"
    m = json.loads(path.read_text())
    base = m["presets"][0]
    m["presets"] = [dict(base, name=key, plate=dict(base["plate"], **plate)) for key, plate in cameras.items()]
    if caps:
        m["caps"] = caps
    path.write_text(json.dumps(m, indent=1))
    return bundle


def bake(bundle, preset, svg):
    run("bake", bundle, "--preset", preset, "-o", svg)
    return svg.read_text()


def inked(svg, px_across):
    """`svg` with every stroke scaled so the heaviest is `HEAVIEST_PX` when its
    viewBox is shown `px_across` pixels wide."""
    across = float(re.search(r'viewBox="([^"]+)"', svg).group(1).split()[2])
    widths = [float(w) for w in re.findall(r'stroke-width="([0-9.e-]+)"', svg)]
    k = HEAVIEST_PX * across / (px_across * max(widths))
    return re.sub(r'stroke-width="([0-9.e-]+)"', lambda m: f'stroke-width="{float(m.group(1)) * k:.6g}"', svg)


def on_paper(svg):
    """The bake's SVG with a ground under it."""
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1)
    x, y, w, h = vb.split()
    rect = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{PAPER}"/>'
    return re.sub(r"(<svg[^>]*>)", r"\1" + rect, svg, count=1)


def grid(svgs, cols, cell=1000, gap=0.04):
    """Tiles at one scale: each bake's drawing centred in a square cell whose
    side is the largest extent among them."""
    parts = []
    for s in svgs:
        x, y, w, h = map(float, re.search(r'viewBox="([^"]+)"', s).group(1).split())
        body = re.search(r"<svg[^>]*>(.*)</svg>", s, re.S).group(1)
        parts.append(((x + w / 2, y + h / 2), max(w, h), body))
    side = max(p[1] for p in parts) * (1 + gap)
    parts = [(c, e, inked(f'<svg viewBox="0 0 {side} {side}">{b}</svg>', README_PX / cols)[len(f'<svg viewBox="0 0 {side} {side}">'):-len("</svg>")])
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
        sys.exit(f"build the CLI first: swift build -c release --package-path KurvenSwift")
    work = Path(tempfile.mkdtemp(prefix="kurven-gallery-"))
    try:
        # Γ, as the published plate frames it: the strip |Im z| ≤ 2.5, a cap
        # per pole band, and that plate's camera.
        gamma = landscape(work, "gamma", "gamma(z)", (-4.5, 4.5), (-2.5, 2.5), 1000, 5,
                          {"plate": dict(shear=-0.5, xAngle=-55.0, zAngle=-90.0, flipX=False)},
                          caps={"kind": "realBands", "beyond": 4.0, "bands": [
                              {"below": -3.5, "cap": 2.5}, {"below": -2.5, "cap": 3.0},
                              {"below": -1.5, "cap": 4.0}, {"below": 0.5, "cap": 5.5}]})
        (OUT / "gamma.svg").write_text(on_paper(inked(bake(gamma, "plate", work / "gamma.svg"), README_PX)))

        # cn(z, 0.64) over the elliptic plate's block of quarter periods,
        # [−K, 5K] × [0, 3K′], with that plate's camera.
        m = 0.64
        K = math.pi / (2 * agm(1, math.sqrt(1 - m)))
        Kp = math.pi / (2 * agm(1, math.sqrt(m)))
        cn = landscape(work, "cn", f"cn(z, {m})", (-K, 5 * K), (0, 3 * Kp), 1200, 4,
                       {"plate": dict(shear=0.51, xAngle=-63.0, zAngle=-90.0, flipX=True)})
        (OUT / "cn.svg").write_text(on_paper(inked(bake(cn, "plate", work / "cn.svg"), README_PX)))

        # 1/(z³ − 1) from four oblique corners: the front face head-on from
        # either side (turn ±90°), the depth receding right or left (shear ±½).
        corners = {f"t{z}s{s}".replace("-", "m"): dict(shear=s, xAngle=-55.0, zAngle=float(z), flipX=True)
                   for z in (-90, 90) for s in (0.5, -0.5)}
        cubic = landscape(work, "cubic", "1/(z^3 - 1)", (-1.3, 1.6), (-1.2, 1.5), 300, 5, corners)
        tiles = [bake(cubic, key, work / f"cubic-{key}.svg") for key in corners]
        (OUT / "cubic-corners.svg").write_text(grid(tiles, 2))
    finally:
        shutil.rmtree(work)
    for f in ("gamma.svg", "cn.svg", "cubic-corners.svg"):
        print(OUT / f, f"{(OUT / f).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
