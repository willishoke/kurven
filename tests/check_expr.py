"""The expression language, held to what it promises.

Three things are being checked, and they fail in different ways. *Parsing* is
checked against hand-written readings of the ambiguous cases -- `-z^2`, `2z^2`,
`1/2z` -- because those are the ones where two reasonable people disagree and
where a port to another language would silently pick the other reading.
*Safety* is checked by showing that the Python that `eval` would have run is
refused by the tokenizer, which is the only claim `kurven.expr` makes about it.
*Numerics* is checked against closed forms where they exist and against an
independently written Euler-Maclaurin summation where they do not -- Borwein's
algorithm and Euler-Maclaurin share no constants and no structure, so agreeing
to ten digits at twenty random points is evidence rather than a tautology.

    python tests/check_expr.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.special as ss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kurven.expr import (  # noqa: E402
    CONSTANTS,
    ExpressionError,
    FUNCTIONS,
    LANGUAGE,
    canonical,
    compile_expression,
    parse,
)

_failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def evaluate(text, z):
    return compile_expression(text)(np.asarray(z, dtype=np.complex128))


def agree(text, other, z, tol=1e-12):
    """Two spellings, one function. Compared where both are finite: the point
    of an expression language is that `1/Γ(z)` and `rgamma(z)` differ at the
    poles and nowhere else, and asserting equality at infinity would be
    asserting something about IEEE rather than about the language."""
    a, b = evaluate(text, z), other
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return False
    return bool(np.max(np.abs(a[both] - b[both])
                       / np.maximum(np.abs(b[both]), 1.0)) < tol)


#: A grid with a pole, a zero, a branch cut and a few generic points on it.
GRID = (np.linspace(-3.5, 3.5, 15)[:, None]
        + 1j * np.linspace(-2.5, 2.5, 13)[None, :])


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def check_precedence():
    z = GRID
    cases = [
        ("-z^2", -(z ** 2), "unary minus is looser than ^"),
        ("2^-z", 2.0 ** (-z), "^ takes a signed exponent"),
        ("2z^2", 2 * z ** 2, "juxtaposition is looser than ^"),
        ("z^2^3", z ** (2 ** 3), "^ is right associative"),
        ("(z^2)^3", (z ** 2) ** 3, "and parenthesizes the other way"),
        ("1/2z", 0.5 * z, "implicit multiplication does not bind tighter"),
        ("z(z+1)", z * (z + 1), "z is not callable, so this is a product"),
        ("(z-1)(z+1)", (z - 1) * (z + 1), "juxtaposed groups multiply"),
        ("2 pi i", 2 * math.pi * 1j * np.ones_like(z), "constants juxtapose"),
        ("2sin(z)", 2 * np.sin(z), "a number times a call"),
        ("1e-3z", 1e-3 * z, "an exponent is part of the number"),
        ("2e", math.e * 2 * np.ones_like(z), "but a bare e is Euler's number"),
        ("z - -z", 2 * z, "a signed right operand"),
        ("--z", z, "and a doubled sign"),
    ]
    for text, want, why in cases:
        check(f"{text!r} reads as {why}", agree(text, want, z))

    check("z! is Γ(z+1)", agree("z!", ss.gamma(z + 1), z))
    check("(z+1)! groups before the factorial",
          agree("(z+1)!", ss.gamma(z + 2), z))
    check("1/Γ(z) is rgamma away from the poles",
          agree("1/Γ(z)", ss.rgamma(z), z, tol=1e-10))
    check("unicode names are the same functions",
          agree("ζ(z) + ψ(z)", evaluate("zeta(z) + digamma(z)", z), z))
    check("aliases are the same functions",
          agree("ln(z) + arcsin(z)", evaluate("log(z) + asin(z)", z), z))


def check_canonical():
    cases = {
        "1 / Γ(z)": "1/gamma(z)",
        "-z^2": "-z^2",
        "2z^2": "2*z^2",
        "z(z+1)": "z*(z+1)",
        "2**z": "2^z",
        "(z-1)(z+1)": "(z-1)*(z+1)",
        "ln(z)": "log(z)",
        "2 PI": None,          # capitalized names are not the same names
    }
    for source, want in cases.items():
        if want is None:
            continue
        got = canonical(source)
        check(f"canonical({source!r}) is {want!r}", got == want, f"got {got!r}")

    stable = True
    same = True
    for source in ["1/Γ(z)", "-z^2+2z-1/(z^2+1)", "cn(z, 0.64)", "z!^2",
                   "2^-z", "exp(1/z)", "(z-1)(z+1)/(z^2+1)"]:
        once = canonical(source)
        stable = stable and canonical(once) == once
        same = same and agree(once, evaluate(source, GRID), GRID)
    check("canonical is idempotent", stable)
    check("and does not change what the expression means", same)


def check_errors():
    cases = [
        ("", "empty input", None),
        ("sin(z", "an unclosed call", 5),
        ("(z+1", "an unclosed group", 4),
        ("z +", "a missing right operand", 3),
        ("foo(z)", "an unknown name", 0),
        ("gamma(z, 2)", "the wrong arity", 0),
        ("z..2", "a malformed number", None),
        ("gamma", "a function used as a value", 0),
        ("z 2 @", "a character the language does not use", 4),
        (")", "a stray bracket", 0),
    ]
    for text, why, position in cases:
        try:
            parse(text)
            check(f"{text!r} is refused: {why}", False, "it parsed")
            continue
        except ExpressionError as e:
            located = position is None or e.position == position
            check(f"{text!r} is refused: {why}", located,
                  f"{e.message} at {e.position}")
        except Exception as e:                                  # noqa: BLE001
            check(f"{text!r} is refused: {why}", False,
                  f"{type(e).__name__} rather than ExpressionError")

    try:
        parse("gama(z)")
        check("a near miss suggests the name it nearly is", False, "it parsed")
    except ExpressionError as e:
        check("a near miss suggests the name it nearly is", "gamma" in e.message,
              e.message)


def check_safety():
    """There is no `eval` behind this, and these are the strings that would
    prove it if there were."""
    hostile = [
        '__import__("os").system("echo pwned")',
        "().__class__.__bases__",
        "z.real",
        "'abc'",
        "lambda: 1",
        "[x for x in (1,2)]",
        "open('/etc/passwd')",
        "z; print(1)",
        "exec('1')",
        "{}.__class__",
    ]
    for text in hostile:
        try:
            parse(text)
            check(f"refuses {text!r}", False, "it parsed")
        except ExpressionError:
            check(f"refuses {text!r}", True)
        except Exception as e:                                  # noqa: BLE001
            check(f"refuses {text!r}", False,
                  f"{type(e).__name__} rather than ExpressionError")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def check_table():
    """Every name the language advertises evaluates. A table entry that raises
    or returns the wrong dtype is a dropdown entry that breaks the app, and the
    only way to know is to call all of them."""
    z = GRID
    broken = []
    for name, f in sorted(FUNCTIONS.items()):
        text = f"{name}(z)" if f.arity == 1 else f"{name}(0.5, z)"
        if name in ("sn", "cn", "dn"):
            text = f"{name}(z, 0.64)"
        try:
            out = evaluate(text, z)
            if out.shape != z.shape or out.dtype != np.complex128:
                broken.append(f"{name}: {out.shape} {out.dtype}")
        except Exception as e:                                  # noqa: BLE001
            broken.append(f"{name}: {type(e).__name__} {e}")
    check(f"all {len(FUNCTIONS)} functions evaluate to a complex grid",
          not broken, "; ".join(broken))

    broken = []
    for name in sorted(CONSTANTS):
        out = evaluate(name, z)
        if out.shape != z.shape or out.dtype != np.complex128:
            broken.append(f"{name}: {out.shape} {out.dtype}")
    check("constants broadcast to the shape of z", not broken, "; ".join(broken))
    check("a constant expression is still a grid",
          evaluate("2i", z).shape == z.shape)
    check("a scalar argument stays scalar",
          evaluate("gamma(z)", 2.0).shape == ())

    named = {f["name"] for f in LANGUAGE["functions"]}
    check("LANGUAGE lists exactly the callable names", named == set(FUNCTIONS),
          f"{len(named)} functions")
    check("LANGUAGE lists exactly the constants",
          {c["name"] for c in LANGUAGE["constants"]} == set(CONSTANTS))
    check("every function documents itself",
          all(f["help"] for f in LANGUAGE["functions"]))


def check_poles_are_values():
    """A pole is `inf`, not an exception: the landscape is a picture of where
    the poles are, so the sampler has to survive standing on one."""
    at_poles = evaluate("gamma(z)", np.array([0.0, -1.0, -2.0], dtype=complex))
    check("Γ at its poles is infinite, not an error",
          bool(np.all(~np.isfinite(at_poles))), str(np.abs(at_poles)))
    check("1/0 is infinite, not an error",
          not np.isfinite(evaluate("1/z", np.array(0j))))
    check("ζ at 1 is infinite, not an error",
          not np.isfinite(evaluate("zeta(z)", np.array(1 + 0j))))
    check("log at 0 is infinite, not an error",
          not np.isfinite(evaluate("log(z)", np.array(0j))))


# --------------------------------------------------------------------------
# zeta
# --------------------------------------------------------------------------


def _zeta_reference(s, terms=14):
    """ζ by Euler-Maclaurin, written here rather than imported.

    This shares nothing with `kurven.expr`'s Borwein path but the answer, which
    is the property that makes it worth writing twice.
    """
    s = np.asarray(s, dtype=np.complex128)
    cut = max(24, int(np.max(np.abs(s.imag))) + 24)
    k = np.arange(1, cut, dtype=float)
    total = np.sum(np.exp(-np.multiply.outer(s, np.log(k))), axis=-1)
    total = total + cut ** (1 - s) / (s - 1) + 0.5 * cut ** (-s)
    bernoulli = ss.bernoulli(2 * terms)
    product = s.copy()
    for j in range(1, terms + 1):
        total = total + (bernoulli[2 * j] / math.factorial(2 * j)
                         * product * cut ** (-s - 2 * j + 1))
        product = product * (s + 2 * j - 1) * (s + 2 * j)
    return total


def check_zeta():
    zeta = compile_expression("zeta(z)")
    known = {
        2.0: math.pi ** 2 / 6,
        0.0: -0.5,
        -1.0: -1.0 / 12.0,
        3.0: 1.2020569031595942,
        0.5: -1.4603545088095868,
        -3.0: 1.0 / 120.0,
        4.0: math.pi ** 4 / 90,
    }
    worst = 0.0
    for s, want in known.items():
        got = complex(zeta(np.array(s, dtype=complex)))
        worst = max(worst, abs(got - want) / max(abs(want), 1.0))
    check("ζ reproduces its closed forms", worst < 1e-12, f"max rel {worst:.2e}")

    trivial = zeta(np.array([-2.0, -4.0, -6.0], dtype=complex))
    check("the trivial zeros are zero", bool(np.max(np.abs(trivial)) < 1e-12),
          f"max |ζ| = {np.max(np.abs(trivial)):.2e}")

    first = zeta(np.array(0.5 + 14.134725141734693j))
    check("the first nontrivial zero is a zero", abs(complex(first)) < 1e-8,
          f"|ζ| = {abs(complex(first)):.2e}")

    # The eta factor `1 - 2^(1-s)` vanishes here, so Borwein's quotient is 0/0
    # and the Euler-Maclaurin fallback is what answers. Without it this point
    # is noise of size 1e10 in a heightfield.
    eta_zero = 1 + 2j * math.pi / math.log(2)
    got = complex(zeta(np.array(eta_zero)))
    want = complex(_zeta_reference(np.array(eta_zero)))
    check("ζ survives the zeros of its own alternating-series factor",
          abs(got - want) / abs(want) < 1e-9,
          f"ζ({eta_zero:.4g}) = {got:.6g} vs {want:.6g}")

    rng = np.random.default_rng(20260919)

    # Right of the critical line both algorithms are well conditioned and the
    # comparison is worth ten digits.
    points = rng.uniform(0.5, 8, 20) + 1j * rng.uniform(-30, 30, 20)
    error = np.max(np.abs(zeta(points) - _zeta_reference(points))
                   / np.abs(_zeta_reference(points)))
    check("ζ agrees with an independent Euler-Maclaurin sum on Re(s) ≥ 1/2",
          error < 1e-11, f"max rel {error:.2e} over 20 points")

    # Left of it the *reference* is the imprecise one, and the looser tolerance
    # here is a statement about Euler-Maclaurin rather than about Borwein. Its
    # direct sum runs to N^|σ| -- 48^4.5 at these points -- and then cancels
    # against N^(1-s)/(s-1), so it loses digits as N grows while the Borwein
    # path does not move. Raising the reference's N makes this number worse,
    # which is the evidence for which side is drifting.
    points = rng.uniform(-5, 0.5, 20) + 1j * rng.uniform(-30, 30, 20)
    error = np.max(np.abs(zeta(points) - _zeta_reference(points))
                   / np.abs(_zeta_reference(points)))
    check("and on Re(s) < 1/2, through the functional equation, to the "
          "reference's own precision", error < 1e-7,
          f"max rel {error:.2e} over 20 points")

    grid = (np.linspace(-5, 8, 500)[:, None]
            + 1j * np.linspace(-30, 30, 400)[None, :])
    started = time.perf_counter()
    values = zeta(grid)
    elapsed = time.perf_counter() - started
    check(f"ζ on {grid.size} points takes {elapsed:.2f} s", elapsed < 3.0,
          f"{np.isfinite(values).mean():.3%} finite")


# --------------------------------------------------------------------------
# elliptic
# --------------------------------------------------------------------------


def check_elliptic():
    z = GRID
    check("cn(z, 0) is cos", agree("cn(z, 0)", np.cos(z), z, tol=1e-10))
    check("sn(z, 0) is sin", agree("sn(z, 0)", np.sin(z), z, tol=1e-10))
    check("dn(z, 0) is 1", agree("dn(z, 0)", np.ones_like(z), z, tol=1e-10))
    check("cn(z, 1) is sech", agree("cn(z, 1)", 1 / np.cosh(z), z, tol=1e-10))

    rng = np.random.default_rng(7)
    w = rng.uniform(-3, 3, 400) + 1j * rng.uniform(-1.2, 1.2, 400)
    for m in (0.2, 0.64, 0.9):
        sn = evaluate(f"sn(z, {m})", w)
        cn = evaluate(f"cn(z, {m})", w)
        dn = evaluate(f"dn(z, {m})", w)
        pythagoras = np.max(np.abs(sn ** 2 + cn ** 2 - 1))
        legendre = np.max(np.abs(dn ** 2 + m * sn ** 2 - 1))
        check(f"sn² + cn² = 1 at m = {m}", pythagoras < 1e-9,
              f"max |Δ| = {pythagoras:.2e}")
        check(f"dn² + m·sn² = 1 at m = {m}", legendre < 1e-9,
              f"max |Δ| = {legendre:.2e}")

    # The elliptic plate's own formula, as the oracle it has been since the
    # notebook: whatever this module does for cn has to be what drew that plate.
    m = 0.64
    u, v = w.real, w.imag
    s_u, c_u, d_u, _ = ss.ellipj(u, m)
    s_v, c_v, d_v, _ = ss.ellipj(v, 1.0 - m)
    plate = ((c_u * c_v - 1j * s_u * d_u * s_v * d_v)
             / (c_v ** 2 + m * s_u ** 2 * s_v ** 2))
    error = np.max(np.abs(evaluate(f"cn(z, {m})", w) - plate))
    check("cn matches examples/elliptic.py's formula", error == 0.0,
          f"max |Δ| = {error:.2e}")


def main():
    print("expression language")
    check_precedence()
    check_canonical()
    check_errors()
    check_safety()
    check_table()
    check_poles_are_values()
    check_zeta()
    check_elliptic()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
