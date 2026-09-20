"""A small language for writing the function down: `1/Γ(z)`, `ζ(z)`, `cn(z, 0.64)`.

Until now a landscape's function was a Python callable chosen by editing an
example. That is fine for four published plates and useless for a dropdown: a
frontend cannot offer what it cannot name, and it cannot accept what it cannot
parse. So the function becomes *text* — a value that travels in a manifest, in
an RPC parameter and in a text field, and that a person can read back.

The obvious way to evaluate that text is `eval`, and the obvious way is wrong.
`eval` accepts the whole of Python from a field a user types into, and the
subset it would need to reject is not expressible as a blocklist. This module
therefore never builds a Python expression at all: it tokenizes, parses into a
tree of six node kinds, and evaluates that tree against a fixed table of
functions. Nothing outside the table is reachable, so nothing outside the table
has to be forbidden.

The grammar is deliberately not Python's. It is the notation the plates are
written in — `z^2`, `2z`, `z!`, `Γ(z)` — and it is small enough to port, which
matters because the Swift side will eventually parse the same strings and must
agree about what `-z^2` means.

    expression := sum
    sum        := product (("+" | "-") product)*
    product    := unary (("*" | "/") unary | unary)*     (* juxtaposition *)
    unary      := ("+" | "-") unary | power
    power      := postfix (("^" | "**") unary)?          (* right associative *)
    postfix    := atom "!"*                              (* n! = Γ(n+1) *)
    atom       := number
                | name
                | name "(" expression ("," expression)* ")"
                | "(" expression ")"
    number     := digit* ["." digit*] [("e" | "E") ["+" | "-"] digit+]
    name       := (letter | "_") (letter | digit | "_")*

Three consequences of those rules worth stating, because they are the ones
people disagree about:

  - `^` binds tighter than unary minus, so `-z^2` is `-(z^2)`. Its right side is
    a `unary`, so `2^-z` parses and `a^b^c` is `a^(b^c)`.
  - Juxtaposition is multiplication at the same precedence as `*`, so `2z^2` is
    `2*(z^2)` and `z(z+1)` is a product — `z` is not callable, and only a name
    in the function table followed by `(` is a call.
  - `1/2z` is `(1/2)*z`, because implicit multiplication is *not* given the
    tighter binding some calculators give it. There is no reading of that
    expression everyone agrees on; this one is at least the same as `1/2*z`.

Evaluation is vectorized and total: it runs under `errstate(all="ignore")`, so a
pole is `inf` and a branch point is `nan` rather than an exception. A landscape
is a picture of where those are, and a sampler that raises at the interesting
points would be a poor instrument.
"""

from __future__ import annotations

import difflib
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.special as ss


class ExpressionError(ValueError):
    """A refusal with a place in the text.

    `position` is a character offset into the source and `length` how much of it
    is at fault, so a client can underline the mistake instead of printing a
    sentence about it. `kind` matches the service's error vocabulary, so the
    same value crosses the RPC boundary without being reclassified.
    """

    kind = "badExpression"

    def __init__(self, message, position=None, length=1):
        super().__init__(message if position is None
                         else f"{message} (at character {position + 1})")
        self.message = message
        self.position = position
        self.length = max(int(length), 1)


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

_OPERATORS = ("**", "+", "-", "*", "/", "^", "!", "(", ")", ",")


@dataclass(frozen=True)
class Token:
    kind: str                 # "number", "name", "op", "end"
    text: str
    position: int

    @property
    def length(self):
        return max(len(self.text), 1)


def _is_name_start(c):
    return c.isalpha() or c == "_"


def _is_name_char(c):
    return c.isalnum() or c == "_"


def tokenize(text):
    """Source to tokens. Knows about numbers, names and the operator set, and
    nothing else — an unexpected character is an error here rather than a
    surprise three stages later."""
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            start = i
            while i < n and text[i].isdigit():
                i += 1
            if i < n and text[i] == ".":
                i += 1
                while i < n and text[i].isdigit():
                    i += 1
            # An `e` is an exponent only when digits follow it; otherwise it is
            # Euler's number, and `2e` is a product.
            if i < n and text[i] in "eE":
                j = i + 1
                if j < n and text[j] in "+-":
                    j += 1
                if j < n and text[j].isdigit():
                    i = j
                    while i < n and text[i].isdigit():
                        i += 1
            tokens.append(Token("number", text[start:i], start))
            continue
        if _is_name_start(c):
            start = i
            while i < n and _is_name_char(text[i]):
                i += 1
            tokens.append(Token("name", text[start:i], start))
            continue
        for op in _OPERATORS:
            if text.startswith(op, i):
                tokens.append(Token("op", op, i))
                i += len(op)
                break
        else:
            raise ExpressionError(f"{c!r} is not something this language uses", i)
    tokens.append(Token("end", "", n))
    return tokens


# --------------------------------------------------------------------------
# the tree
# --------------------------------------------------------------------------
#
# Six node kinds, each with `evaluate` (against a complex array) and `format`
# (back to canonical source). Keeping the two next to each other is what keeps
# `parse(canonical(x))` meaning what `x` meant: a printing rule that disagreed
# with the parsing rule would show up as a node that prints one way and reads
# another.

#: Precedence levels, for deciding where `format` must put parentheses.
_SUM, _PRODUCT, _UNARY, _POWER, _POSTFIX, _ATOM = 1, 2, 3, 4, 5, 6


def _parenthesize(text, context, mine):
    """Parenthesize when the node binds more loosely than the place it is
    printed into — the one rule, stated once, that keeps `format` the inverse
    of the parser instead of an approximation to it."""
    return f"({text})" if mine < context else text


@dataclass(frozen=True)
class Num:
    value: float

    def evaluate(self, z):
        return np.complex128(self.value)

    def format(self, context=_SUM):
        v = self.value
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)


@dataclass(frozen=True)
class Var:
    def evaluate(self, z):
        return z

    def format(self, context=_SUM):
        return "z"


@dataclass(frozen=True)
class Const:
    name: str
    value: complex

    def evaluate(self, z):
        return np.complex128(self.value)

    def format(self, context=_SUM):
        return self.name


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple

    def evaluate(self, z):
        return FUNCTIONS[self.name].fn(*(a.evaluate(z) for a in self.args))

    def format(self, context=_SUM):
        return f"{self.name}({', '.join(a.format() for a in self.args)})"


@dataclass(frozen=True)
class BinOp:
    op: str
    left: object
    right: object

    def evaluate(self, z):
        a, b = self.left.evaluate(z), self.right.evaluate(z)
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b
        return a ** b

    def format(self, context=_SUM):
        if self.op == "^":
            # Right associative: the left side of a power needs parentheses
            # when it is itself a power, the right side never does.
            text = f"{self.left.format(_POSTFIX)}^{self.right.format(_UNARY)}"
            return _parenthesize(text, context, _POWER)
        mine = _SUM if self.op in "+-" else _PRODUCT
        text = (f"{self.left.format(mine)}{self.op}"
                f"{self.right.format(mine + 1)}")
        return _parenthesize(text, context, mine)


@dataclass(frozen=True)
class Neg:
    operand: object

    def evaluate(self, z):
        return -self.operand.evaluate(z)

    def format(self, context=_SUM):
        return _parenthesize(f"-{self.operand.format(_UNARY)}", context, _UNARY)


@dataclass(frozen=True)
class Factorial:
    operand: object

    def evaluate(self, z):
        return ss.gamma(self.operand.evaluate(z) + 1)

    def format(self, context=_SUM):
        return f"{self.operand.format(_POSTFIX)}!"


# --------------------------------------------------------------------------
# the parser
# --------------------------------------------------------------------------


class _Parser:
    def __init__(self, text):
        self.text = text
        self.tokens = tokenize(text)
        self.at = 0

    # -- token plumbing

    @property
    def token(self):
        return self.tokens[self.at]

    def advance(self):
        token = self.tokens[self.at]
        self.at += 1
        return token

    def looking_at(self, *texts):
        return self.token.kind == "op" and self.token.text in texts

    def expect(self, text, what):
        if not self.looking_at(text):
            raise ExpressionError(f"expected {text!r} {what}",
                                  self.token.position, self.token.length)
        return self.advance()

    # -- the grammar

    def parse(self):
        if self.token.kind == "end":
            raise ExpressionError("there is no expression here, only empty space", 0)
        node = self.sum()
        if self.token.kind != "end":
            raise ExpressionError(
                f"{self.token.text!r} has nothing to attach to",
                self.token.position, self.token.length)
        return node

    def sum(self):
        node = self.product()
        while self.looking_at("+", "-"):
            op = self.advance().text
            node = BinOp(op, node, self.product())
        return node

    def product(self):
        node = self.unary()
        while True:
            if self.looking_at("*", "/"):
                op = self.advance().text
                node = BinOp(op, node, self.unary())
            elif self.starts_an_atom():
                # Juxtaposition. The right side is a full `unary` so that `2z^2`
                # is `2*(z^2)` rather than `(2*z)^2`.
                node = BinOp("*", node, self.unary())
            else:
                return node

    def starts_an_atom(self):
        return (self.token.kind in ("number", "name")
                or (self.token.kind == "op" and self.token.text == "("))

    def unary(self):
        if self.looking_at("-"):
            self.advance()
            return Neg(self.unary())
        if self.looking_at("+"):
            self.advance()
            return self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.looking_at("^", "**"):
            self.advance()
            return BinOp("^", base, self.unary())
        return base

    def postfix(self):
        node = self.atom()
        while self.looking_at("!"):
            self.advance()
            node = Factorial(node)
        return node

    def atom(self):
        token = self.token
        if token.kind == "number":
            self.advance()
            return Num(float(token.text))
        if token.kind == "name":
            return self.name(self.advance())
        if self.looking_at("("):
            self.advance()
            node = self.sum()
            self.expect(")", "to close the group")
            return node
        if token.kind == "end":
            raise ExpressionError("the expression stops in the middle",
                                  token.position)
        raise ExpressionError(f"{token.text!r} cannot start a term",
                              token.position, token.length)

    def name(self, token):
        name = CANONICAL_NAMES.get(token.text)
        if name is None:
            raise ExpressionError(self._unknown(token.text), token.position,
                                  token.length)
        if name in FUNCTIONS:
            if not self.looking_at("("):
                raise ExpressionError(
                    f"{name} is a function; it needs an argument, as in "
                    f"{name}({'z, 0.5' if FUNCTIONS[name].arity == 2 else 'z'})",
                    token.position, token.length)
            self.advance()
            args = [self.sum()]
            while self.looking_at(","):
                self.advance()
                args.append(self.sum())
            close = self.expect(")", f"to close the call to {name}")
            wanted = FUNCTIONS[name].arity
            if len(args) != wanted:
                raise ExpressionError(
                    f"{name} takes {wanted} argument{'' if wanted == 1 else 's'}, "
                    f"not {len(args)}",
                    token.position, close.position - token.position + 1)
            return Call(name, tuple(args))
        if name == "z":
            return Var()
        return Const(name, CONSTANTS[name].value)

    def _unknown(self, text):
        close = difflib.get_close_matches(text, sorted(CANONICAL_NAMES), n=3,
                                          cutoff=0.6)
        if close:
            return f"there is no {text!r}; did you mean {' or '.join(close)}?"
        return (f"there is no {text!r} in this language; the variable is z, and "
                f"`describe` lists what else there is")


# --------------------------------------------------------------------------
# the function table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Function:
    name: str
    arity: int
    fn: object
    help: str
    aliases: tuple = ()


@dataclass(frozen=True)
class Constant:
    name: str
    value: complex
    help: str
    aliases: tuple = ()


def _parameter(value):
    """A second argument that is a *parameter*, not a point of the plane.

    A Bessel order and an elliptic modulus index a family of functions; the
    landscape is a function of z alone. scipy's `jv` and `ellipj` refuse a
    complex one, and continuing them in that argument is not something this
    language is promising, so the imaginary part is dropped rather than
    honoured. A scalar stays a scalar so that scipy takes its fast path.
    """
    value = np.real(np.asarray(value, dtype=np.complex128))
    return float(np.ravel(value)[0]) if value.size == 1 else value


def _complex(values):
    """A real result, as complex. `abs`, `re`, `im` and `arg` are not
    holomorphic and so cannot be landscapes of anything interesting on their
    own, but they are useful inside one (`abs(z)^2`), and everything in the
    tree has to have one dtype."""
    return np.asarray(values).astype(np.complex128)


# -- zeta ------------------------------------------------------------------
#
# scipy's zeta is real-only, and mpmath's is a scalar loop: a 200k-point grid
# through mpmath is minutes. Borwein's alternating series gives every point of
# the grid at once in a few dozen vectorized passes, which is the difference
# between a dropdown entry and a cache file someone has to generate first.

_LOG_GROWTH = math.log(3 + math.sqrt(8))


@lru_cache(maxsize=8)
def _borwein_coefficients(n):
    """`d_k` for k = 0..n, by the ratio between consecutive terms.

    The closed form is a sum of factorials that overflows well before the
    `n` this needs; the ratio `4(n+j)(n-j)/((2j+1)(2j+2))` is the same sequence
    without ever forming one.
    """
    d = np.empty(n + 1)
    term, total = 1.0 / n, 0.0
    for j in range(n + 1):
        total += term
        d[j] = n * total
        term *= 4.0 * (n + j) * (n - j) / ((2 * j + 1) * (2 * j + 2))
    return d


def _borwein_order(t_max):
    """How many terms the error bound needs at this height.

    `|γ_n(s)| ≲ 3(1 + 2|t|)e^{π|t|/2}(3+√8)^{-n}`, so the cost is linear in
    |Im s| — which is why the order is chosen from the grid rather than fixed:
    the ζ plate runs to |t| = 30 and a disc of radius 2 around the origin does
    not, and neither should pay the other's price.
    """
    need = (math.pi * t_max / 2 + math.log(3.0 * (1 + 2 * t_max))
            + 15 * math.log(10)) / _LOG_GROWTH
    return int(min(max(math.ceil(need), 16), 200))


def _zeta_euler_maclaurin(s, terms=12):
    """ζ by Euler–Maclaurin, for the handful of points Borwein cannot do.

    Borwein computes η(s) and divides by `1 - 2^(1-s)`, which vanishes at
    `s = 1 + 2πik/ln 2` — points on the line σ = 1 at heights 9.06, 18.13,
    27.19, right where the ζ landscape is interesting. η vanishes there too, so
    the quotient is 0/0 and the answer is noise rather than a number. Those
    points are rare enough to be worth a second, slower algorithm and too
    visible to leave as spikes in the heightfield.
    """
    s = np.asarray(s, dtype=np.complex128)
    cut = max(10, int(np.max(np.abs(s.imag))) + 10) if s.size else 10
    k = np.arange(1, cut, dtype=float)
    total = np.sum(np.exp(-np.multiply.outer(s, np.log(k))), axis=-1)
    total = total + cut ** (1 - s) / (s - 1) + 0.5 * cut ** (-s)
    bernoulli = ss.bernoulli(2 * terms)
    product = s.copy()                      # (s)(s+1)...(s+2j-2), j = 1 so far
    for j in range(1, terms + 1):
        total = total + (bernoulli[2 * j] / math.factorial(2 * j)
                         * product * cut ** (-s - 2 * j + 1))
        product = product * (s + 2 * j - 1) * (s + 2 * j)
    return total


def _zeta_halfplane(s):
    """ζ on Re(s) ≥ 1/2, by Borwein's algorithm."""
    n = _borwein_order(float(np.max(np.abs(s.imag))) if s.size else 0.0)
    d = _borwein_coefficients(n)
    # Scaled by d_n, so every term is in [-1, 0] and the alternating sum loses
    # no more than the handful of digits its own length costs.
    scaled = (d[:n] - d[n]) / d[n]
    total = np.zeros(s.shape, dtype=np.complex128)
    for k in range(n):
        weight = scaled[k] if k % 2 == 0 else -scaled[k]
        total += weight * np.exp(-s * math.log(k + 1))
    eta = 1.0 - 2.0 ** (1.0 - s)
    out = -total / eta
    poor = np.abs(eta) < 1e-5
    if poor.any():
        out[poor] = _zeta_euler_maclaurin(s[poor])
    return out


def _zeta(s):
    """The Riemann zeta function of a complex array.

    Re(s) < 1/2 goes through the functional equation, which needs ζ of the
    reflected point — where Re > 1/2 and the series converges quickly.
    """
    s = np.asarray(s, dtype=np.complex128)
    flat = np.ravel(s)
    out = np.empty(flat.shape, dtype=np.complex128)
    left = flat.real < 0.5
    right = ~left
    if right.any():
        out[right] = _zeta_halfplane(flat[right])
    if left.any():
        r = flat[left]
        out[left] = (2.0 ** r * np.pi ** (r - 1) * np.sin(np.pi * r / 2)
                     * ss.gamma(1 - r) * _zeta_halfplane(1 - r))
    return out.reshape(np.shape(s))


# -- Jacobi elliptic functions ---------------------------------------------


def _jacobi(z, m):
    """sn, cn and dn of a complex argument, from the real ones.

    scipy's `ellipj` is real-only; the imaginary direction comes from the
    addition formulas with the complementary parameter `1 - m`. `m` is a
    parameter, not a point of the plane, so its imaginary part is dropped
    rather than honoured — there is no continuation here to be faithful to.
    """
    z = np.asarray(z, dtype=np.complex128)
    parameter = _parameter(m)
    u, v = z.real, z.imag
    s, c, d, _ = ss.ellipj(u, parameter)
    s1, c1, d1, _ = ss.ellipj(v, 1.0 - parameter)
    denominator = c1 ** 2 + parameter * s ** 2 * s1 ** 2
    sn = (s * d1 + 1j * c * d * s1 * c1) / denominator
    cn = (c * c1 - 1j * s * d * s1 * d1) / denominator
    dn = (d * c1 * d1 - 1j * parameter * s * c * s1) / denominator
    return sn, cn, dn


#: Everything the language can call. The table is the whitelist: a name that is
#: not here is not reachable, which is why there is nothing to forbid.
_FUNCTION_LIST = [
    Function("exp", 1, np.exp, "e raised to z"),
    Function("log", 1, np.log, "natural logarithm, principal branch", ("ln",)),
    Function("sqrt", 1, np.sqrt, "square root, principal branch"),
    Function("sin", 1, np.sin, "sine"),
    Function("cos", 1, np.cos, "cosine"),
    Function("tan", 1, np.tan, "tangent — poles at the odd multiples of π/2"),
    Function("cot", 1, lambda z: 1.0 / np.tan(z), "cotangent"),
    Function("sec", 1, lambda z: 1.0 / np.cos(z), "secant"),
    Function("csc", 1, lambda z: 1.0 / np.sin(z), "cosecant"),
    Function("sinh", 1, np.sinh, "hyperbolic sine"),
    Function("cosh", 1, np.cosh, "hyperbolic cosine"),
    Function("tanh", 1, np.tanh, "hyperbolic tangent"),
    Function("asin", 1, np.arcsin, "inverse sine", ("arcsin",)),
    Function("acos", 1, np.arccos, "inverse cosine", ("arccos",)),
    Function("atan", 1, np.arctan, "inverse tangent", ("arctan",)),
    Function("asinh", 1, np.arcsinh, "inverse hyperbolic sine", ("arcsinh",)),
    Function("acosh", 1, np.arccosh, "inverse hyperbolic cosine", ("arccosh",)),
    Function("atanh", 1, np.arctanh, "inverse hyperbolic tangent", ("arctanh",)),
    Function("gamma", 1, ss.gamma, "Γ(z) — poles at the non-positive integers",
             ("Γ",)),
    Function("rgamma", 1, ss.rgamma, "1/Γ(z), entire; zeros at 0, -1, -2, ..."),
    Function("loggamma", 1, ss.loggamma, "log Γ(z), principal branch",
             ("lgamma",)),
    Function("digamma", 1, ss.psi, "ψ(z) = Γ'(z)/Γ(z)", ("psi", "ψ")),
    Function("zeta", 1, _zeta, "ζ(z) — the pole is at z = 1", ("ζ",)),
    Function("erf", 1, ss.erf, "the error function"),
    Function("erfc", 1, ss.erfc, "the complementary error function"),
    Function("erfi", 1, ss.erfi, "the imaginary error function"),
    Function("wofz", 1, ss.wofz, "the Faddeeva function w(z)", ("faddeeva",)),
    Function("expi", 1, lambda z: -ss.exp1(-z),
             "the exponential integral, continued as -E₁(-z)", ("Ei",)),
    Function("lambertw", 1, lambda z: ss.lambertw(z),
             "W(z), the principal branch of the Lambert W function", ("W",)),
    Function("airyai", 1, lambda z: ss.airy(z)[0], "Ai(z)", ("Ai",)),
    Function("airybi", 1, lambda z: ss.airy(z)[2], "Bi(z)", ("Bi",)),
    Function("besselj", 2, lambda n, z: ss.jv(_parameter(n), z),
             "J_n(z), first argument the order", ("jv",)),
    Function("bessely", 2, lambda n, z: ss.yv(_parameter(n), z),
             "Y_n(z), first argument the order", ("yv",)),
    Function("besseli", 2, lambda n, z: ss.iv(_parameter(n), z),
             "I_n(z), first argument the order", ("iv",)),
    Function("besselk", 2, lambda n, z: ss.kv(_parameter(n), z),
             "K_n(z), first argument the order", ("kv",)),
    Function("sn", 2, lambda z, m: _jacobi(z, m)[0],
             "Jacobi sn(z, m), doubly periodic"),
    Function("cn", 2, lambda z, m: _jacobi(z, m)[1],
             "Jacobi cn(z, m), doubly periodic"),
    Function("dn", 2, lambda z, m: _jacobi(z, m)[2],
             "Jacobi dn(z, m), doubly periodic"),
    Function("abs", 1, lambda z: _complex(np.abs(z)), "|z|, as a real value"),
    Function("re", 1, lambda z: _complex(np.real(z)), "the real part"),
    Function("im", 1, lambda z: _complex(np.imag(z)), "the imaginary part"),
    Function("conj", 1, np.conj, "the complex conjugate"),
    Function("arg", 1, lambda z: _complex(np.angle(z)), "the argument of z"),
]

_CONSTANT_LIST = [
    Constant("i", 1j, "the imaginary unit", ("j",)),
    Constant("pi", math.pi, "π", ("π",)),
    Constant("e", math.e, "Euler's number"),
    Constant("tau", 2 * math.pi, "2π", ("τ",)),
]

FUNCTIONS = {f.name: f for f in _FUNCTION_LIST}
CONSTANTS = {c.name: c for c in _CONSTANT_LIST}

#: Every spelling, mapped to the one the tree stores. Aliases exist so that a
#: person can type what they would write on paper (`Γ`, `ln`, `arcsin`) and a
#: manifest can still record one name for one function.
CANONICAL_NAMES = {"z": "z"}
for _f in _FUNCTION_LIST:
    CANONICAL_NAMES[_f.name] = _f.name
    for _alias in _f.aliases:
        CANONICAL_NAMES[_alias] = _f.name
for _c in _CONSTANT_LIST:
    CANONICAL_NAMES[_c.name] = _c.name
    for _alias in _c.aliases:
        CANONICAL_NAMES[_alias] = _c.name

#: What a client needs to build a form and a help panel, as data. The frontend
#: does not get a second, hand-written copy of this list to fall out of step
#: with; it asks.
LANGUAGE = {
    "variable": "z",
    "constants": [{"name": c.name, "aliases": list(c.aliases), "help": c.help}
                  for c in _CONSTANT_LIST],
    "functions": [{"name": f.name, "arity": f.arity, "aliases": list(f.aliases),
                   "help": f.help} for f in _FUNCTION_LIST],
    "operators": ["+", "-", "*", "/", "^", "**", "!"],
}


# --------------------------------------------------------------------------
# the public three
# --------------------------------------------------------------------------


def parse(text):
    """Text to tree, or an `ExpressionError` that says where."""
    return _Parser(text).parse()


def compile_expression(text):
    """Text to a vectorized `f(z)`.

    The result always has `z`'s shape and complex dtype, including for an
    expression that does not mention `z` — a constant landscape is flat, not
    scalar, and the caller should not have to know which it asked for.
    """
    node = parse(text)

    def f(z):
        z = np.asarray(z, dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                out = np.asarray(node.evaluate(z), dtype=np.complex128)
        if out.shape == z.shape:
            return out
        return np.broadcast_to(out, z.shape).astype(np.complex128, copy=True)

    f.expression = node.format()
    f.__doc__ = f"f(z) = {f.expression}"
    return f


def canonical(text):
    """The same expression, spelled one way.

    Aliases resolve (`Γ` → `gamma`), implicit multiplication becomes explicit,
    `**` becomes `^`, and spacing goes away. It is what a manifest records, so
    that two bundles of the same function compare equal, and it is idempotent —
    the output parses to a tree that prints as itself.
    """
    return parse(text).format()
