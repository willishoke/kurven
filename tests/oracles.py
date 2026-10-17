"""Write the oracle fixtures: answers kurven did not compute.

Python is in this repository for one purpose. Where the Swift package
implements something a third-party library already implements, that library is
the oracle, and this script asks it for the answers and writes them to
`tests/fixtures`. Three sets:

    expr/     every function of the expression language on a grid that runs
              along the real axis at exactly im = 0 (the branch cuts), plus each
              special function on the window a landscape of it uses. numpy for
              the elementary functions, scipy.special where it has a complex
              form, mpmath where it does not. Every function in the language
              table is named in `ORACLES` with the library that answers it; a
              function with no oracle is refused here, not skipped.
    contour/  contourpy's iso-lines on three awkward grids -- nested loops,
              nothing but saddles, and a step function -- so the native
              marching squares can be held to them.
    npy/      numpy's own writer: one file per dtype the reader must accept,
              and two it must reject (Fortran order, an unsupported dtype).

Everything else under `tests/fixtures` is golden, written by Swift with
`kurven-cli fixtures`, and is not this script's business.

The files are committed and CI never runs this. Run it by hand when an oracle
or a case changes, and commit what it writes as its own change:

    uv run tests/oracles.py

Determinism: contouring runs single-chunk, the point sets come from a seeded
generator, and the contoured grids are float32 because that is what the
Swift side reads out of a bundle.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import contourpy
import mpmath
import numpy as np
import scipy.special as ss

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def save(path, array):
    """`np.save`, C-contiguous: the Swift reader refuses Fortran order rather
    than read it transposed, so a fixture must say what it means."""
    np.save(path, np.ascontiguousarray(array))


# --------------------------------------------------------------------------
# expr: the oracle for every function in the language
# --------------------------------------------------------------------------

mpmath.mp.dps = 30


def _mp(fn):
    """A scalar mpmath function over a complex array, at 30 digits, rounded
    once to complex128. A pole mpmath reports as an infinity stays one."""
    def lifted(z):
        flat = np.ravel(np.asarray(z, dtype=np.complex128))
        out = np.empty(flat.shape, dtype=np.complex128)
        for k, w in enumerate(flat):
            try:
                out[k] = complex(fn(mpmath.mpc(w.real, w.imag)))
            except (ZeroDivisionError, ValueError):
                out[k] = complex(np.inf, 0.0)
        return out.reshape(np.shape(z))
    return lifted


def _jacobi(kind):
    """sn, cn or dn at parameter m (not modulus): mpmath's `ellipfun`, which
    takes a complex argument where scipy's `ellipj` takes only a real one."""
    return lambda z, m: _mp(lambda w: mpmath.ellipfun(kind, w, m=m))(z)


def _real(fn):
    return lambda z: np.asarray(fn(z)).astype(np.complex128)


#: Every function in the language table, with the library that answers it and
#: the call. One-argument functions take z; the Bessel family takes (order, z)
#: and the Jacobi three (z, m), with the parameter a real scalar.
ORACLES = {
    "exp": ("numpy", np.exp),
    "log": ("numpy", np.log),
    "sqrt": ("numpy", np.sqrt),
    "sin": ("numpy", np.sin),
    "cos": ("numpy", np.cos),
    "tan": ("numpy", np.tan),
    "cot": ("numpy", lambda z: 1.0 / np.tan(z)),
    "sec": ("numpy", lambda z: 1.0 / np.cos(z)),
    "csc": ("numpy", lambda z: 1.0 / np.sin(z)),
    "sinh": ("numpy", np.sinh),
    "cosh": ("numpy", np.cosh),
    "tanh": ("numpy", np.tanh),
    "asin": ("numpy", np.arcsin),
    "acos": ("numpy", np.arccos),
    "atan": ("numpy", np.arctan),
    "asinh": ("numpy", np.arcsinh),
    "acosh": ("numpy", np.arccosh),
    "atanh": ("numpy", np.arctanh),
    "abs": ("numpy", _real(np.abs)),
    "re": ("numpy", _real(np.real)),
    "im": ("numpy", _real(np.imag)),
    "conj": ("numpy", np.conj),
    "arg": ("numpy", _real(np.angle)),
    "gamma": ("scipy.special.gamma", ss.gamma),
    "rgamma": ("scipy.special.rgamma", ss.rgamma),
    "loggamma": ("scipy.special.loggamma", ss.loggamma),
    "digamma": ("scipy.special.psi", ss.psi),
    "erf": ("scipy.special.erf", ss.erf),
    "erfc": ("scipy.special.erfc", ss.erfc),
    "erfi": ("scipy.special.erfi", ss.erfi),
    "wofz": ("scipy.special.wofz", ss.wofz),
    # Ei(z) = -E1(-z): the language's branch, which scipy's own complex `expi`
    # does not share on the negative real axis.
    "expi": ("scipy.special.exp1", lambda z: -ss.exp1(-z)),
    "lambertw": ("scipy.special.lambertw", lambda z: ss.lambertw(z)),
    "airyai": ("scipy.special.airy", lambda z: ss.airy(z)[0]),
    "airybi": ("scipy.special.airy", lambda z: ss.airy(z)[2]),
    "besselj": ("scipy.special.jv", ss.jv),
    "bessely": ("scipy.special.yv", ss.yv),
    "besseli": ("scipy.special.iv", ss.iv),
    "besselk": ("scipy.special.kv", ss.kv),
    # scipy's zeta and ellipj are real-only.
    "zeta": ("mpmath.zeta", _mp(mpmath.zeta)),
    "sn": ("mpmath.ellipfun", _jacobi("sn")),
    "cn": ("mpmath.ellipfun", _jacobi("cn")),
    "dn": ("mpmath.ellipfun", _jacobi("dn")),
}

#: Functions held to a functional equation in `Expressions.swift` instead of
#: to an oracle, because no library here has a complex form of them. Empty,
#: and a name may only move here with that suite written.
UNHELD: dict[str, str] = {}

TWO_ARGUMENT = {"besselj", "bessely", "besseli", "besselk"}
JACOBI = {"sn", "cn", "dn"}

#: The Bessel and Airy engines are not AMOS, and agree with it to about 1e-11
#: across these windows; the rest run the oracle's own method, or one as good.
LOOSE = {"besselj", "bessely", "besseli", "besselk", "airyai", "airybi", "zeta"}

GRID = {"real": [-3.5, 3.5, 15], "imag": [-2.5, 2.5, 13]}

#: The special functions on the windows their landscapes use, and a little
#: beyond: the regimes each implementation switches between are in here.
#: (name, function, parameter, tolerance, real, imag)
WIDE = [
    ("gamma_wide", "gamma", None, 1e-12, [-12, 12, 49], [-6, 6, 25]),
    ("rgamma_wide", "rgamma", None, 1e-12, [-12, 12, 49], [-6, 6, 25]),
    ("loggamma_wide", "loggamma", None, 1e-12, [-12, 12, 49], [-6, 6, 25]),
    ("digamma_wide", "digamma", None, 1e-12, [-12, 12, 49], [-6, 6, 25]),
    ("zeta_wide", "zeta", None, 1e-10, [-5, 8, 27], [-30, 30, 31]),
    ("erf_wide", "erf", None, 1e-12, [-8, 8, 33], [-8, 8, 33]),
    ("erfc_wide", "erfc", None, 1e-12, [-8, 8, 33], [-8, 8, 33]),
    ("erfi_wide", "erfi", None, 1e-12, [-8, 8, 33], [-8, 8, 33]),
    ("wofz_wide", "wofz", None, 1e-12, [-8, 8, 33], [-8, 8, 33]),
    ("expi_wide", "expi", None, 1e-10, [-25, 25, 51], [-25, 25, 51]),
    ("lambertw_wide", "lambertw", None, 1e-10, [-8, 8, 33], [-8, 8, 33]),
    ("besselj0_wide", "besselj", 0, 1e-9, [-25, 25, 51], [-8, 8, 17]),
    ("besselj10_wide", "besselj", 10, 1e-9, [-30, 30, 31], [-30, 30, 31]),
    ("bessely2p7_wide", "bessely", 2.7, 1e-9, [-30, 30, 31], [-30, 30, 31]),
    ("besseli1p5_wide", "besseli", 1.5, 1e-9, [-25, 25, 51], [-8, 8, 17]),
    ("besselk5_wide", "besselk", 5, 1e-9, [-30, 30, 31], [-30, 30, 31]),
    ("besselkm2p3_wide", "besselk", -2.3, 1e-9, [-25, 25, 26], [-25, 25, 26]),
    ("airyai_wide", "airyai", None, 1e-9, [-12, 12, 49], [-12, 12, 49]),
    ("airybi_wide", "airybi", None, 1e-9, [-12, 12, 49], [-12, 12, 49]),
    ("cn_wide", "cn", 0.9, 1e-12, [-6, 6, 25], [-4, 4, 17]),
    ("sn_wide", "sn", 0.2, 1e-12, [-6, 6, 25], [-4, 4, 17]),
    ("tan_wide", "tan", None, 1e-12, [-12, 12, 49], [-4, 4, 17]),
    ("sinh_wide", "sinh", None, 1e-12, [-12, 12, 49], [-12, 12, 49]),
]


def _spelling(name, parameter):
    """The expression the Swift side compiles for this case."""
    if name in JACOBI:
        return f"{name}(z, {parameter})"
    if name in TWO_ARGUMENT:
        return f"{name}({parameter}, z)"
    return f"{name}(z)"


def _answer(name, parameter, z):
    library, fn = ORACLES[name]
    with np.errstate(all="ignore"):
        if name in JACOBI:
            values = fn(z, float(parameter))
        elif name in TWO_ARGUMENT:
            values = fn(float(parameter), z)
        else:
            values = fn(z)
    return library, np.asarray(values, dtype=np.complex128)


def expr_fixtures(out, language):
    """Every function of the language on the table grid, and the special ones
    on their wide windows. `language` is the frozen table the Swift suites hold
    the evaluator's own table to; every name in it must have an oracle here."""
    names = [f["name"] for f in language["functions"]]
    missing = sorted(set(names) - set(ORACLES) - set(UNHELD))
    if missing:
        raise SystemExit(f"no oracle for {', '.join(missing)}: name one in ORACLES, "
                         "or hold it to a functional equation and list it in UNHELD")
    stale = sorted((set(ORACLES) | set(UNHELD)) - set(names))
    if stale:
        raise SystemExit(f"oracles for functions the language does not have: {stale}")

    cases = []

    def case(stem, name, parameter, tol, real, imag):
        r = np.linspace(real[0], real[1], real[2])
        i = np.linspace(imag[0], imag[1], imag[2])
        library, values = _answer(name, parameter, r[:, None] + 1j * i[None, :])
        save(out / f"{stem}.npy", values)
        cases.append({"name": stem, "expression": _spelling(name, parameter),
                      "oracle": library, "real": real, "imag": imag,
                      "file": f"{stem}.npy", "tol": tol})

    for name in sorted(ORACLES):
        parameter = 0.64 if name in JACOBI else 0.5 if name in TWO_ARGUMENT else None
        case(f"table_{name}", name, parameter, 1e-9 if name in LOOSE else 1e-12,
             GRID["real"], GRID["imag"])
    for stem, name, parameter, tol, real, imag in WIDE:
        case(stem, name, parameter, tol, real, imag)

    (out / "index.json").write_text(json.dumps({
        "grid": GRID,
        "columns": "[real][imag]",
        "cases": cases,
        "unheld": UNHELD,
    }, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(cases)


# --------------------------------------------------------------------------
# contour
# --------------------------------------------------------------------------


def contour_fixtures(out):
    """contourpy's answer on grids chosen to exercise the awkward cases.

    Three fields, contoured at levels that are deliberately not round numbers
    relative to the data:

      ripple   smooth, many nested closed loops, contours that leave the grid
      saddles  a product of sines: nothing but saddle points, which is the one
               case marching squares has to make a choice about
      cliff    a step function, so contours run along cell boundaries and every
               interpolation lands at an endpoint

    Stored as one flat array per level plus path offsets, in domain
    coordinates, columns (imag, real) as contourpy returns them for an array
    indexed [real][imag].
    """
    n = 61
    real_bounds, imag_bounds = (-3.0, 3.0), (-2.0, 2.0)
    real = np.linspace(*real_bounds, n)
    imag = np.linspace(*imag_bounds, n)
    R, I = np.meshgrid(real, imag, indexing="ij")

    fields = {
        "ripple": np.sin(1.7 * R) * np.cos(2.3 * I) + 0.35 * R,
        "saddles": np.sin(3.0 * R) * np.sin(3.0 * I),
        "cliff": np.where(R + 0.5 * I > 0.3, 1.0, -1.0) + 0.001 * I,
    }
    levels = {
        "ripple": [-0.9, -0.31, 0.0, 0.17, 0.62, 1.05],
        "saddles": [-0.5, -0.02, 0.0, 0.02, 0.5],
        "cliff": [0.0, 0.5],
    }

    index = {}
    for name, field in fields.items():
        # Contour the float32 array: the Swift side reads float32 out of the
        # bundle, and two implementations interpolating between different
        # numbers disagree for a reason that has nothing to do with either.
        field = field.astype(np.float32)
        save(out / f"{name}.npy", field.T)                      # (ny, nx), world order
        generator = contourpy.contour_generator(
            z=field, name="threaded", line_type=contourpy.LineType.Separate,
            chunk_count=1)
        for lvl in levels[name]:
            segs = []
            for seg in generator.lines(float(lvl)):
                if len(seg) < 2:
                    continue
                # Grid index to domain: n samples span n - 1 intervals.
                xy = seg.copy()
                xy[:, 0] *= (imag_bounds[1] - imag_bounds[0]) / (n - 1)
                xy[:, 0] += imag_bounds[0]
                xy[:, 1] *= (real_bounds[1] - real_bounds[0]) / (n - 1)
                xy[:, 1] += real_bounds[0]
                segs.append(xy)
            tag = f"{name}_{lvl}".replace("-", "m").replace(".", "p")
            if segs:
                save(out / f"{tag}.npy", np.concatenate(segs))
                save(out / f"{tag}.idx.npy",
                     np.concatenate([[0], np.cumsum([len(s) for s in segs])]).astype(np.int64))
            else:
                save(out / f"{tag}.npy", np.zeros((0, 2)))
                save(out / f"{tag}.idx.npy", np.zeros(1, dtype=np.int64))
            index.setdefault(name, []).append({"level": lvl, "tag": tag,
                                               "paths": len(segs)})

    (out / "index.json").write_text(json.dumps({
        "grid": {"nx": n, "ny": n, "real": list(real_bounds), "imag": list(imag_bounds)},
        "fields": index,
        "columns": ["imag", "real"],
    }, sort_keys=True, indent=1) + "\n")


# --------------------------------------------------------------------------
# npy
# --------------------------------------------------------------------------


def npy_fixtures(out):
    """One file per dtype the reader must accept, and two it must reject with
    a typed error rather than a wrong answer."""
    rng = np.random.default_rng(7)
    save(out / "f4.npy", rng.normal(size=(3, 5)).astype(np.float32))
    save(out / "f8.npy", rng.normal(size=(4, 2)).astype(np.float64))
    save(out / "i8.npy", rng.integers(-2**40, 2**40, size=(6,)).astype(np.int64))
    save(out / "c16.npy",
         (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))).astype(np.complex128))
    save(out / "f8_1d.npy", np.arange(7, dtype=np.float64))
    # Rejected: Fortran order, and an unsupported dtype.
    np.save(out / "reject_fortran.npy", np.asfortranarray(rng.normal(size=(3, 3))))
    save(out / "reject_dtype.npy", rng.integers(0, 200, size=(4,)).astype(np.uint8))


# --------------------------------------------------------------------------


def _fresh(path, keep=()):
    """Empty `path` of everything but `keep`, which belongs to the golden
    writer."""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name in keep:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def main():
    expr = FIXTURES / "expr"
    language = json.loads((expr / "language.json").read_text(encoding="utf-8"))["language"]
    # expr/ is shared: the frozen language and the golden landscape bundles
    # live beside the oracle's values.
    _fresh(expr, keep={"language.json", "landscapes.json"}
           | {p.name for p in expr.glob("*.kurven")})
    for sub in ("contour", "npy"):
        _fresh(FIXTURES / sub)

    cases = expr_fixtures(expr, language)
    contour_fixtures(FIXTURES / "contour")
    npy_fixtures(FIXTURES / "npy")
    print(f"wrote the oracle sets under {FIXTURES.relative_to(ROOT)}: "
          f"expr ({cases} cases), contour, npy")


if __name__ == "__main__":
    main()
