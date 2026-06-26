# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Frozen-dataclass AST nodes for the expression engine (TODO 17 + 18).

In-node evaluation: each node implements evaluate(mode) and stores its own
result (TODO 18); the lexer/parser arrive with item 20. Number lexemes are
the UNSIGNED raw source text — sign is a UnaryOp; literal well-formedness is
the lexer's contract, not validated here.
"""

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from mcp_abacus.expr.value import (
    EvalContext,
    InexactHandling,
    Mode,
    NotRepresentableError,
    UndefinedVariableError,
    Value,
    VariableStore,
    _lexeme_scale,
)

UNARY_OPS: frozenset[str] = frozenset({"+", "-", "~"})  # ~ is bitwise NOT (24.3.2)
# ** is POWER; ^ & | are bitwise (24.3.2 — ^ is XOR, not power).
BINARY_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", "//", "%", "**", "&", "|", "^"})

# 26.8: the analyze-tree label spells each operator out as an UPPER_SNAKE opcode
# (BINARY_ADD, UNARY_NEG) instead of the bare symbol, so the printout reads as a
# named operation. One entry per operator in the frozensets above — a missing
# entry is a hard KeyError when the tree renders, never a silent fallback.
_UNARY_OPCODES: dict[str, str] = {"+": "UNARY_POS", "-": "UNARY_NEG", "~": "UNARY_NOT"}
_BINARY_OPCODES: dict[str, str] = {
    "+": "BINARY_ADD",
    "-": "BINARY_SUB",
    "*": "BINARY_MUL",
    "/": "BINARY_DIV",
    "//": "BINARY_FLOORDIV",
    "%": "BINARY_MOD",
    "**": "BINARY_POW",
    "&": "BINARY_AND",
    "|": "BINARY_OR",
    "^": "BINARY_XOR",
}

# Value has no operator dunders by design (19.5) — dispatch to its named methods.
_UNARY_FUNCS: dict[str, Callable[[Value], Value]] = {
    "+": Value.pos,
    "-": Value.neg,
    "~": Value.bitnot,
}
_BINARY_FUNCS: dict[str, Callable[[Value, Value], Value]] = {
    "+": Value.add,
    "-": Value.sub,
    "*": Value.mul,
    "/": Value.div,
    "//": Value.floordiv,
    "%": Value.mod,
    "**": Value.pow,  # our ** is power, routed to Value.pow — never a Python dunder (18.2)
    "&": Value.bitand,
    "|": Value.bitor,
    "^": Value.bitxor,  # ^ is XOR (24.3.2), routed to Value.bitxor — never a dunder (18.2)
}

# The function set (22.3): name -> the per-mode Value method it dispatches to
# (19.5/22.4), parallel to the operator tables above. THIS registry is the single
# source — adding a function is one entry here plus its Value method, the same
# two-step shape as adding an operator. The function set is not the lexer's or
# parser's concern; both consult this table (the parser via FUNCTION_ARITIES).
_FUNCS: dict[str, Callable[..., Value]] = {
    "abs": Value.abs_,  # 22.4.1 — exact magnitude in every mode (shape of neg())
    "conj": Value.conj,  # 40.12 — complex conjugate a-bi; identity on reals
    "re": Value.re,  # 40.12 — real part Re(z); identity on reals
    "im": Value.im,  # 40.12 — imaginary part Im(z); 0 on reals
    "arg": Value.arg,  # 40.12 — argument/phase atan2(Im, Re); 0/pi on reals
    "sign": Value.sign,  # 40.9 — signum -1/0/+1; exact classification (float keeps binary64 flag)
    "sqrt": Value.sqrt,  # 22.4.2 — irrational, inexact except on the mode's grid
    "cbrt": Value.cbrt,  # 28.21 — cube root; odd root so negatives OK, inexact except perfect cubes
    "hypot": Value.hypot,  # 40.20 — variadic Euclidean norm sqrt(Σ xi**2); inherits sqrt's stance
    "pow": Value.pow,  # 28.20 — binary; the call form of **, reuses Value.pow (fixed-arity 2)
    "floor": Value.floor,  # 28.23 — round toward -inf; optional ndigits (1,2), mostly exact
    "ceil": Value.ceil,  # 28.24 — round toward +inf; mirror of floor, optional ndigits (1,2)
    "round": Value.round_,  # 28.25 — round to nearest, ties to even; optional ndigits (1,2)
    "trunc": Value.trunc,  # 28.26 — round toward zero (drop the fraction); optional ndigits (1,2)
    "sin": Value.sin,  # 28.10 — transcendental; fixed-point Taylor series, else inexact/refuse
    "cos": Value.cos,  # 28.11 — sin's machinery; even Taylor series, else inexact/refuse
    "tan": Value.tan,  # 28.12 — sin/cos; fixed-point divides the two series, else inexact/refuse
    "cot": Value.cot,  # 28.13 — cos/sin; mirror of tan, undefined where sin = 0, else inexact
    "asin": Value.asin,  # 28.14 — arcsine; atan(x/sqrt(1-x^2)), domain |x|<=1, else inexact/refuse
    "acos": Value.acos,  # 28.15 — arccosine; pi/2 - asin(x), domain |x|<=1, else inexact/refuse
    "atan": Value.atan,  # 28.16 — arctangent; the arctan series itself, all reals, else inexact
    "atan2": Value.atan2,  # 40.1 — binary; quadrant-aware angle of (x, y), else inexact/refuse
    "degrees": Value.degrees,  # 40.11 — radians -> degrees, x*180/pi; unit scaling, else inexact
    "radians": Value.radians,  # 40.11 — degrees -> radians, x*pi/180; unit scaling, else inexact
    "log": Value.log,  # 28.17/40.10 — NATURAL log unary; log(x, base) two-arg general log
    "ln": Value.ln,  # 28.17 — natural-log-only wrapper kept UNARY (log overloads, ln does not)
    "log10": Value.log10,  # 28.18 — base-10 log; ln(x)/ln(10), exact on powers of ten
    "log2": Value.log2,  # 28.19 — base-2 log; ln(x)/ln(2), inexact except log2(1)=0
    "exp": Value.exp,  # 28.27 — e**x; reduce by ln2 + exp series, inexact/refuse off the grid
    "sinh": Value.sinh,  # 40.2 — (e**x - e**-x)/2 via the exp core, inexact except sinh(0)=0
    "cosh": Value.cosh,  # 40.2 — (e**x + e**-x)/2 via the exp core, inexact except cosh(0)=1
    "tanh": Value.tanh,  # 40.2 — sinh/cosh via the exp core, inexact except tanh(0)=0
    "asinh": Value.asinh,  # 40.3 — ln(x+sqrt(x^2+1)) via ln, all reals, except asinh(0)=0
    "acosh": Value.acosh,  # 40.3 — ln(x+sqrt(x^2-1)) via ln, domain x>=1, except acosh(1)=0
    "atanh": Value.atanh,  # 40.3 — ln((1+x)/(1-x))/2 via ln, domain |x|<1, except atanh(0)=0
    "avg": Value.avg,  # 28.4 — variadic; sum / count, follows the mode's / rule
    "max": Value.max_,  # 28.2 — variadic; selection (largest), exact, carries the operand verbatim
    "min": Value.min_,  # 28.3 — variadic; mirror of max (smallest)
    "median": Value.median,  # 28.7 — variadic; order-only, odd selects (exact), even averages
    "quantile": Value.quantile,  # 40.12 — (q, data…); type-7 order statistic, q in [0,1]
    "percentile": Value.percentile,  # 40.12 — (p, data…); quantile scaled by 100, p in [0,100]
    "clamp": Value.clamp,  # 40.21 — ternary; min(hi, max(lo, x)) selection, exact, refuses lo>hi
    "lerp": Value.lerp,  # 40.22 — ternary; linear interpolation a+(b-a)*t, arithmetic stance
    "variance": Value.variance,  # 28.8 — variadic; population sum-of-squared-deviations / n
    "stddev": Value.stddev,  # 28.9 — variadic; sqrt(variance), inherits sqrt's per-mode story
    "covariance": Value.covariance,  # 40.13 — two vectors; population mean((x-mx)*(y-my))
    "correlation": Value.correlation,  # 40.14 — two vectors; Pearson r, cov/(stddev*stddev)
    "gcd": Value.gcd,  # 40.7 — variadic; math.gcd of magnitudes, integer-only, exact everywhere
    "lcm": Value.lcm,  # 40.8 — variadic; math.lcm of magnitudes, integer-only, zero absorbs to 0
    "factorial": Value.factorial,  # 40.4 — n! for a non-negative integer, exact every mode, capped
    "comb": Value.comb,  # 40.5 — binary; binomial coefficient C(n, k), exact integer every mode
    "perm": Value.perm,  # 40.6 — binary; k-permutations P(n, k), exact integer every mode
    "factor": Value.factor,  # 40.23 — unary; prime factors as a VECTOR (the first vector producer)
    "pct": Value.pct,  # 36.1 — p percent of x (x*p/100), follows the mode's / rule
    "pct_change": Value.pct_change,  # 36.1 — signed relative change (new-old)/old, mode's / rule
    "bps": Value.bps,  # 36.2 — b basis points of x (x*b/10000), follows the mode's / rule
    "compound": Value.compound,  # 36.3 — principal*(1+rate)**periods, per-period rate, mode's rule
    "pmt": Value.pmt,  # 36.4 — annuity payment amortising pv over nper at per-period rate
    "fv": Value.fv,  # 36.4 — future value of an nper-period payment stream at per-period rate
    "pv": Value.pv,  # 36.4 — present value of an nper-period payment stream at per-period rate
}

# Functions that ACCEPT a vector operand (19.1.10). The blanket vector refusal in
# FuncCall._evaluate exempts these names, letting the Value method see the VECTOR
# payload and reduce over its elements; everything else still refuses a vector. The
# whole variadic stats family opts in: the selection aggregates max/min (28.2/28.3)
# and the computing ones avg/median/variance/stddev (28.4/28.7/28.8/28.9) take a
# single vector via _series_operands; covariance/correlation (40.13/40.14) take TWO
# vectors directly. The integer-only reducers gcd/lcm (40.7/40.8) take a single
# vector through the same _series_operands path.
_VECTOR_FUNCS: frozenset[str] = frozenset(
    {
        "max",
        "min",
        "avg",
        "median",
        "quantile",
        "percentile",
        "variance",
        "stddev",
        "covariance",
        "correlation",
        "gcd",
        "lcm",
    }
)

# The nullary set (29.2): zero-argument calls like pi(). A SECOND registry, parallel
# to _FUNCS but a different callable KIND — these are NOT operand-methods. With no
# operand to carry the mode, a nullary takes the per-run EvalContext (29.1) and
# returns a Value; FuncCall dispatches to it with the context instead of evaluated
# operands. Same single-source rule as _FUNCS: a nullary is one entry here plus its
# Value constructor. Arity is a fixed 0 (set in FUNCTION_ARITIES), NOT read off the
# signature — the ctx parameter is engine-injected, never a user argument.
_NULLARY_FUNCS: dict[str, Callable[[EvalContext], Value]] = {
    "pi": Value.pi,  # 29.2 — circle constant; per-mode value 29.3
    "e": Value.e,  # 29.2 — Euler's number; per-mode value 29.3
    "time": Value.time,  # 28.1 — Unix epoch; reads ctx.now_ns, exact in all but float
}


# The special-form set (40.18): a THIRD callable KIND beside _FUNCS and _NULLARY_FUNCS.
# A special form does NOT receive evaluated operands — it takes the run EvalContext and
# its arguments' UNEVALUATED subtrees, because (like the solver) some arguments are an
# unevaluated sub-program and a free variable NAME, not values. ``integral(expr, var, a,
# b)`` is the first: ``expr`` is re-evaluated at many sample points with ``var`` bound to
# each, never walked once as an operand. The implementation needs the AST (Node.evaluate),
# which value.py cannot see, so unlike every other function it lives in the sibling engine
# module expr.forms, NOT as a Value method; FuncCall._evaluate defers the import to it (the
# same call-time-import trick reference.py uses for the solver, avoiding a load-time cycle).
# Only the static name -> arity table lives here, the single source the parser validates
# against; arity is fixed per form (not read off a signature — the args are Nodes, not the
# call's user arguments in the operand-method sense).
_SPECIAL_FORM_ARITIES: dict[str, tuple[int, int | None]] = {
    "integral": (4, 4),  # 40.18 — integral(expr, var, a, b); definite integral, always inexact
    "diff": (3, 3),  # 40.17 — diff(expr, var, at); numerical derivative, always inexact
    "sum": (4, 4),  # 40.19 — sum(i, lo, hi, expr); range summation Σ, EXACT finite fold
    "product": (4, 4),  # 40.19 — product(i, lo, hi, expr); range product Π, EXACT finite fold
}
_SPECIAL_FORM_NAMES: frozenset[str] = frozenset(_SPECIAL_FORM_ARITIES)

# Which ARGUMENT of each special form is the bound variable NAME (the dummy the form
# rebinds per sample/term), so it is masked from referenced_names rather than leaking as a
# free solver unknown. ``integral``/``diff`` put the integrand FIRST (var at index 1);
# ``sum``/``product`` (40.19) follow the Σ/Π reading order ``sum(i, lo, hi, expr)`` — the
# index NAME comes first (index 0), the body last.
_SPECIAL_FORM_BOUND_VAR: dict[str, int] = {
    "integral": 1,
    "diff": 1,
    "sum": 0,
    "product": 0,
}


# The nullaries that double as bare constants (29.6): pi and e may be written WITHOUT
# parentheses, so `2*pi` reads the constant like a literal. The parser turns a bare
# `pi`/`e` into the same nullary FuncCall as `pi()`/`e()`, so evaluation is identical.
# time() is NOT here — it reads the clock, an action that stays an explicit call. These
# names are also reserved: assigning to them (`pi = ...`) is a parse error. Every name
# here MUST be a key of _NULLARY_FUNCS (a constant is a nullary that omits its parens).
CONSTANT_NAMES: frozenset[str] = frozenset({"pi", "e"})


# One-line semantics per callable (21.2.3), the help-tool counterpart to the
# registries above: the inline comments next to _FUNCS/_NULLARY_FUNCS are code,
# not data, so the reference cannot render them — this table promotes them to
# strings the `functions` section reads. Same single-source rule as the registries:
# every name in _FUNCS and _NULLARY_FUNCS (incl. the `ln` alias) MUST appear here,
# enforced by a reference test, so the help cannot drift from what is wired. Each
# value is the descriptor AFTER the rendered signature — facts only, no signature
# (the section prepends it); the model fills in the prose.
FUNCTION_HELP: dict[str, str] = {
    "abs": "absolute value (complex: modulus sqrt(re^2+im^2)); exact in every type",
    "conj": "complex conjugate a-bi; the identity on real values",
    "re": "real part Re(z); the identity on real values",
    "im": "imaginary part Im(z); 0 on real values",
    "arg": "argument/phase in radians, atan2(Im, Re); 0 or pi on real values",
    "sign": "signum -1/0/+1 by the operand's sign; an exact classification (works on any value), "
    "float keeps the binary64 inexact flag",
    "sqrt": "square root; refuses negatives, inexact except on the type's grid "
    "(rational needs a perfect square)",
    "cbrt": "cube root; negatives OK (odd root), inexact except on the type's grid "
    "(rational needs a perfect cube)",
    "hypot": "variadic euclidean norm sqrt(x1^2+...+xn^2); any reals, inexact except "
    "on the type's grid (rational needs a perfect-square sum)",
    "pow": "x to the power y; the call form of the ** operator",
    "floor": "round toward -inf; optional ndigits (default 0); exact except float with ndigits>0",
    "ceil": "round toward +inf; optional ndigits (default 0); exact except float with ndigits>0",
    "round": "round to nearest, ties to even; optional ndigits (default 0); "
    "exact except float with ndigits>0",
    "trunc": "round toward zero (drop the fraction); optional ndigits (default 0); "
    "exact except float with ndigits>0",
    "sin": "sine, radians; inexact except sin(0)=0, rational refuses non-zero",
    "cos": "cosine, radians; inexact except cos(0)=1, rational refuses non-zero",
    "tan": "tangent, radians; inexact except tan(0)=0, undefined at odd multiples of pi/2",
    "cot": "cotangent, radians; always inexact, undefined at multiples of pi (incl. 0)",
    "asin": "arcsine, radians in [-pi/2, pi/2]; domain |x|<=1, inexact except asin(0)=0",
    "acos": "arccosine, radians in [0, pi]; domain |x|<=1, inexact except acos(1)=0",
    "atan": "arctangent, radians in (-pi/2, pi/2); all reals, inexact except atan(0)=0",
    "atan2": "two-arg arctangent; quadrant angle of (x,y) in (-pi,pi], inexact off atan2(0,x>=0)=0",
    "degrees": "radians to degrees, x*180/pi; inexact except 0, rational refuses x!=0",
    "radians": "degrees to radians, x*pi/180; inexact except 0, rational refuses x!=0",
    "log": "natural log base e; the two-arg log(x, base) is the general logarithm "
    "log(x)/log(base), exact only on integer powers; refuses x<=0 or base<=0/base=1",
    "ln": "natural log; the strictly unary spelling of log(x)",
    "log10": "base-10 log; exact on powers of ten, inexact otherwise, refuses x<=0",
    "log2": "base-2 log; inexact except log2(1)=0, refuses x<=0",
    "exp": "exponential e**x, inverse of log; inexact except exp(0)=1, rational refuses non-zero",
    "sinh": "hyperbolic sine (e**x-e**-x)/2; inexact except sinh(0)=0, rational refuses non-zero",
    "cosh": "hyperbolic cosine (e**x+e**-x)/2; inexact except cosh(0)=1, rational refuses non-zero",
    "tanh": "hyperbolic tangent, range (-1,1); inexact except tanh(0)=0, refuses non-zero rational",
    "asinh": "inverse hyperbolic sine, ln(x+sqrt(x^2+1)); all reals, inexact except asinh(0)=0",
    "acosh": "inverse hyperbolic cosine, ln(x+sqrt(x^2-1)); domain x>=1, inexact except acosh(1)=0",
    "atanh": "inverse hyperbolic tangent, ln((1+x)/(1-x))/2; |x|<1, inexact except atanh(0)=0",
    "sum": "range summation Σ: fold the body over the index NAMED by the 1st arg from the "
    "2nd arg to the 3rd INCLUSIVE (integer steps; 1st arg a bare name, 4th the unevaluated "
    "body — NOT values); repeated +, EXACT in every type, capped at 100000 terms",
    "product": "range product Π: fold the body over the index NAMED by the 1st arg from the "
    "2nd arg to the 3rd INCLUSIVE (integer steps; 1st arg a bare name, 4th the unevaluated "
    "body — NOT values); repeated *, may round in fixed-point/float, capped at 100000 terms",
    "avg": "arithmetic mean, sum/count of the operands (or a single vector's elements); "
    "follows the type's / rule",
    "max": "largest operand (or element of a single vector), returned verbatim; exact",
    "min": "smallest operand (or element of a single vector), returned verbatim; exact",
    "median": "middle operand by value (or of a single vector's elements); odd count exact, "
    "even averages the two middles",
    "quantile": "value at quantile fraction q in [0,1] of the data (a run or one vector), "
    "type-7 linear; exact on a datum, else interpolates (rational exact, fixed-point/float round)",
    "percentile": "value at percentile rank p in [0,100] of the data; quantile scaled by 100, "
    "so percentile(50, …) is the median; exact on a datum, else interpolates",
    "clamp": "constrain x to [lo, hi] = min(hi, max(lo, x)); selection not math, exact in every "
    "type, carries the chosen operand verbatim, refuses lo>hi",
    "lerp": "linear interpolation a+(b-a)*t; exact in rational, may round in fixed-point/float, "
    "t unrestricted (outside [0,1] extrapolates)",
    "variance": "population variance, sum of squared deviations / n of the operands (or a "
    "single vector's elements)",
    "stddev": "population standard deviation, sqrt of variance (operands or a single vector)",
    "covariance": "population covariance of two equal-length vectors, mean((x-mx)*(y-my)); "
    "exact in rational, may round in fixed-point/float",
    "correlation": "Pearson correlation of two equal-length vectors, cov/(stddev*stddev) in "
    "[-1, 1]; inherits stddev's sqrt (always inexact in float/fixed-point, rational refuses an "
    "irrational root), undefined for a constant series",
    "gcd": "greatest common divisor of the operands; integer-only, sign-dropped, exact everywhere",
    "lcm": "least common multiple of the operands; integer-only, any zero gives 0, exact always",
    "factorial": "n! for a non-negative integer; exact in every type, refuses negative/non-integer "
    "operands, capped at 1000 (float refuses past the double range, ~n>170)",
    "comb": "binomial coefficient n!/(k!(n-k)!) — count of k-subsets of n; integer-only, exact in "
    "every type, k<0 or k>n is 0, non-integer args refuse (the gamma extension)",
    "perm": "falling factorial n!/(n-k)! — count of ordered k-permutations of n; integer-only, "
    "exact in every type, k<0 or k>n is 0, non-integer args refuse (the gamma extension)",
    "factor": "prime factors as an ascending vector with multiplicity (factor(12)=[2,2,3], "
    "factor(1)=[]); positive integer only, exact in fixed-point/rational, capped at 10**12",
    "pct": "p percent of x (x*p/100); explicit so the caller never hand-rolls /100, follows the "
    "type's / rule",
    "pct_change": "signed relative change (new-old)/old as a fraction; follows the type's / rule, "
    "divides by zero when old is 0",
    "bps": "b basis points of x (x*b/10000); the bps-vs-percent-safe twin of pct, follows the "
    "type's / rule",
    "compound": "compound growth principal*(1+rate)**periods; rate is PER PERIOD and periods "
    "counts the same unit, follows the type's power/multiply rules",
    "pmt": "ordinary-annuity payment pv*r/(1-(1+r)**-nper) amortising pv to zero (pv/nper when "
    "r=0); rate per period, no sign flip, follows the type's / rule",
    "fv": "ordinary-annuity future value pmt*((1+r)**nper-1)/r of a payment stream (pmt*nper when "
    "r=0); rate per period, follows the type's / rule",
    "pv": "ordinary-annuity present value pmt*(1-(1+r)**-nper)/r of a payment stream (pmt*nper "
    "when r=0); rate per period, follows the type's / rule",
    "pi": "circle constant pi, usable bare as `pi`; inexact in fixed-point/float, rational refuses",
    "e": "Euler's number e, usable bare as `e`; inexact in fixed-point/float, rational refuses",
    "time": "current Unix epoch seconds; exact except in float",
    "integral": "definite integral of an expression over [a, b] w.r.t. the variable NAMED by "
    "the 2nd arg (1st arg is the unevaluated integrand, 2nd a bare name — NOT values); "
    "adaptive-Simpson quadrature, always inexact",
    "diff": "numerical derivative of an expression at a point w.r.t. the variable NAMED by "
    "the 2nd arg (1st arg is the unevaluated expression, 2nd a bare name, 3rd the point — "
    "NOT values); five-point central difference, always inexact",
}


def _arity_of(func: Callable[..., Value]) -> tuple[int, int | None]:
    """Allowed argument count (min, max) READ OFF a method's signature (22.2).

    self IS the first operand, so a unary method like sqrt(self) is (1, 1) and a
    binary one (self, other) is (2, 2). A ``*args`` (VAR_POSITIONAL) tail makes the
    max unbounded — sum_(self, *others) is (1, None) — so variadic funcs declare a
    MINIMUM, not a fixed count. A positional param that carries a DEFAULT is OPTIONAL
    (22.8): it lifts the max but not the min, so round(self, ndigits=None) reads
    (1, 2) — a required operand plus one optional trailing arg (the consuming method
    reads that arg as a Python int; the count is not validated here). Reading it off
    the method keeps it from drifting from _FUNCS, the same invariant the single-count
    form had.
    """
    params = inspect.signature(func).parameters.values()
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in positional if p.default is p.empty)
    variadic = any(p.kind is p.VAR_POSITIONAL for p in params)
    return (required, None if variadic else len(positional))


# Arity range per function (min, max|None). The parser (22.2) and FuncCall both
# validate a call's argument count against this without importing the methods.
# Nullaries (29.2) join with a fixed (0, 0) — their ctx parameter is engine-
# injected, so the count is NOT read off the signature like the operand-methods.
FUNCTION_ARITIES: dict[str, tuple[int, int | None]] = {
    **{name: _arity_of(func) for name, func in _FUNCS.items()},
    **{name: (0, 0) for name in _NULLARY_FUNCS},
    **_SPECIAL_FORM_ARITIES,  # 40.18 — fixed per form, not read off a signature
}


def _describe_arity(lo: int, hi: int | None) -> str:
    """Human phrasing of an arity range, for the wrong-count error messages."""
    if hi is None:
        return f"at least {lo} argument(s)"
    if hi == lo:
        return f"{lo} argument(s)"
    return f"{lo} to {hi} argument(s)"


def _arity_ok(name: str, count: int) -> bool:
    """Whether ``count`` arguments is allowed for the registered function ``name``."""
    lo, hi = FUNCTION_ARITIES[name]
    return lo <= count and (hi is None or count <= hi)


class EvalError(Exception):
    """Evaluation failure, carrying the 1-based source line of the failing node."""

    def __init__(self, message: str, line: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


def _validate_line(line: int) -> None:
    if line < 1:
        raise ValueError(f"line must be >= 1, got {line}")


def _parenthesize(node: "Node") -> str:
    """A child node's source, wrapped in parens when it could re-associate (35.2.2).

    Used by the source() unparse to keep a reconstructed sub-expression
    unambiguous: a binary op or an assignment nested inside another op gets
    parentheses (``a * (b + c)``), an atom (a literal, a variable, a call) does not.
    """
    rendered = node.source()
    return f"({rendered})" if isinstance(node, (BinOp, Assign)) else rendered


def _operand_value_string(node: "Node", scale: int | None) -> str:
    """An already-evaluated node rendered as its VALUE at the active precision (35.3.1).

    The abort headline shows operands as the NUMBERS they evaluated to, not the
    lexemes the user typed: ``scale`` is the result's fixed-point precision, so a
    literal ``3`` beside a scale-2 quotient reads as ``3.00``. Only ever called from
    the abort path, after the walk has stored every operand's value — so the value is
    present; the assert records that invariant.
    """
    assert node.value is not None
    return node.value.to_string(scale)


def _abort_message(node: "Node", value: Value) -> str:
    """The abort-on-inexact diagnostic for the node whose value first went inexact (35.3).

    A headline followed by hint lines, one per line. The headline (35.3.1) always
    reads the same: ``Inexact calculation in line X: <operands> = <result> is not
    exact.`` — the operands and result are the computed NUMBERS at the active
    precision (the result's fixed-point scale), never the source lexemes, so a
    ``1 / 3`` written by the caller shows as ``1.00 / 3.00 = 0.33`` once the active
    precision is two decimals. Composed where EVERYTHING about the inexactness is in
    hand: the node lays out the operation (``_operand_form``) and carries, via the
    EvalError the caller wraps this in, the source line.

    The first hint (35.3.2) always follows: how to LIFT the abort — switch to the
    ``continue-and-report`` policy that computes the inexact result instead of
    rejecting it. The remaining, CONDITIONAL hints — the residual / raise-precision
    steer, the inexact-function note, the rational-mode-is-exact offer
    (``explain_inexact`` has them ready) — are deferred to 35.3.3-35.3.5.
    """
    scale = value.precision()  # the active fixed-point precision; None outside fixed-point
    headline = (
        f"Inexact calculation in line {node.line}: "
        f"{node._operand_form(scale)} = {value.to_string(scale)} is not exact."
    )
    enable = InexactHandling.CONTINUE_AND_REPORT.value  # the policy that lifts the abort
    hints = [f" - Pass inexact_handling='{enable}' to enable inexact calculations."]
    return "\n".join([headline, *hints])


class Node(ABC):
    """Abstract base of all AST nodes."""

    __slots__ = ()

    # Declared here for the shared evaluate()/pretty() machinery; every
    # concrete subclass defines both as dataclass fields.
    line: int
    value: Value | None

    @abstractmethod
    def _label(self) -> str:
        """One-line node header: kind + op/quoted-lexeme (no line number — 26.4)."""

    @abstractmethod
    def source(self) -> str:
        """Reconstruct this node as a readable infix source string (35.2.2).

        An unparse — not necessarily byte-identical to the original text (spacing is
        normalised, redundant grouping may differ), but a faithful, re-parseable
        rendering of the sub-expression. Nested binary ops/assignments are
        parenthesized via ``_parenthesize`` so the reading stays unambiguous. The
        abort-on-inexact headline lays the operation out in VALUES instead (see
        ``_operand_form``); this stays the lexeme-level view for the analyze tree.
        """

    def _operand_form(self, scale: int | None) -> str:
        """This node's operation with each operand shown as its computed VALUE (35.3.1).

        The abort headline reads ``<operands> = <result>`` in the NUMBERS the walk
        produced, not source lexemes — ``scale`` is the result's active fixed-point
        precision, so a literal ``3`` beside a scale-2 result shows as ``3.00``.
        Compound nodes override to interleave their operator / call syntax with the
        children's values; the default — an atom (a literal or a variable) — has no
        operands to lay out, so it is just its own value.
        """
        return _operand_value_string(self, scale)

    @abstractmethod
    def _children(self) -> tuple["Node", ...]:
        """Child nodes in source order."""

    @abstractmethod
    def _evaluate(self, ctx: EvalContext) -> Value:
        """Compute this node's Value under the run context (no storing, no wrapping).

        Children recurse through ``_walk`` so the SAME context flows down the whole
        tree; ``ctx`` carries the mode and the fixed-point precision floor (29.1).
        """

    def evaluate(
        self,
        mode: Mode,
        min_fixed_point_precision: int = 0,
        now_ns: int | None = None,
        variables: VariableStore | None = None,
        inexact_handling: InexactHandling = InexactHandling.CONTINUE_AND_REPORT,
    ) -> Value:
        """Evaluate the subtree in ONE mode; store and return this node's Value.

        Re-evaluating under another mode OVERWRITES the stored values (18.5).
        Arithmetic failures raise EvalError carrying the failing node's line.

        ``min_fixed_point_precision`` (25.2.1) is the fixed-point scale floor;
        like ``mode`` it is threaded down and only consumed on the literals, where
        it raises each fixed-point operand to at least that many decimals so the
        scale propagates through the calculation. Defaults to 0 (no floor); it is
        a no-op outside fixed-point mode, which has no decimal scale.

        ``now_ns`` (28.1.2) is the single realtime clock reading ``time()`` renders;
        sampled ONCE here so every ``time()`` in the run sees one instant. Defaults
        to the real clock (``time.time_ns()``); tests pass a fixed epoch to assert
        exact per-mode/scale renders.

        ``inexact_handling`` (35.2) is the caller's policy when a result is inexact:
        the default CONTINUE_AND_REPORT computes and lets the verdict surface, while
        ABORT_ON_INEXACT raises an EvalError — tagged with the offending node's line
        and naming the sub-expression and the kind/magnitude of the inexactness — the
        moment any value in the walk is inexact. It rides the EvalContext like
        ``mode``; see ``_walk`` for the check.

        ``variables`` (31.7) SEEDS the run's VariableStore with bindings set before
        the walk — the solver pre-binds the unknown to a candidate value so the
        program reads it like any other name, while the program's own assignments
        fill in the remaining constants into the same store. Defaults to None, a
        fresh empty store: a bare ``calculate`` run starts with no bindings, exactly
        as before. The store is used as given (not copied), so a caller reusing one
        across runs sees the run's assignments accumulate — the solver passes a
        fresh store per candidate to keep evaluations independent.

        This is the public entry: it bundles the run state into the per-run
        EvalContext (29.1) and walks the tree threading that one object down,
        rather than passing the state to every node or reaching for a module global.
        In FIXED_POINT it first PRE-WALKS the tree once to derive the nullary scale
        (29.3) — the floor raised to the widest literal scale — so a nullary like
        ``pi()``, which has no operand to carry a scale, is computed to match the
        precision of the literals it shares the expression with.
        """
        nullary_precision = min_fixed_point_precision
        if mode in (Mode.FIXED_POINT, Mode.COMPLEX):  # both carry a fixed-point scale
            nullary_precision = max(min_fixed_point_precision, self._max_literal_scale())
        ctx = EvalContext(
            mode=mode,
            min_fixed_point_precision=min_fixed_point_precision,
            nullary_precision=nullary_precision,
            now_ns=time.time_ns() if now_ns is None else now_ns,
            variables=variables if variables is not None else VariableStore(),
            inexact_handling=inexact_handling,
        )
        return self._walk(ctx)

    def _max_literal_scale(self) -> int:
        """Largest written decimal scale among the literals in this subtree (29.3).

        The pre-walk behind ``nullary_precision``: every node folds the max over its
        children, ``Number`` overriding with its own lexeme scale. 0 for a subtree
        with no literal (e.g. a bare ``pi()``), leaving the nullary scale at the floor.
        """
        return max((child._max_literal_scale() for child in self._children()), default=0)

    def referenced_names(self) -> frozenset[str]:
        """Names READ as variable references (Var) anywhere in this subtree (31.4).

        Folds the union over the children; ``Var`` overrides to contribute its own
        name. An ``Assign`` target is NOT a reference — it is reported by
        ``assigned_names`` instead — so an assignment's name appears here only if the
        name is also read somewhere (e.g. on a right-hand side).
        """
        return frozenset().union(*(child.referenced_names() for child in self._children()))

    def assigned_names(self) -> frozenset[str]:
        """Names BOUND as assignment targets (Assign) anywhere in this subtree (31.4).

        Folds the union over the children; ``Assign`` overrides to add its own target
        name. Used to tell a free unknown (only read) from a computed constant (bound
        by an assignment) when validating a solver request.
        """
        return frozenset().union(*(child.assigned_names() for child in self._children()))

    def _walk(self, ctx: EvalContext) -> Value:
        """Evaluate this node under a shared context; store and return its Value.

        The recursion step every node funnels through (children call it on their
        own children), so one EvalContext built at the top reaches the whole walk.
        """
        try:
            result = self._evaluate(ctx)
        except ArithmeticError as exc:
            # A child's EvalError is not ArithmeticError, so it propagates
            # untouched — the line stays the innermost failing node's (18.4).
            raise EvalError(str(exc), line=self.line) from exc
        object.__setattr__(self, "value", result)  # designated slot on a frozen node (18.5)
        if ctx.inexact_handling is InexactHandling.ABORT_ON_INEXACT and not result.exact:
            # ABORT_ON_INEXACT (35.2.2): unwind the moment a value is inexact. Because
            # the walk is depth-first (children walk before the parent), THIS is the
            # FIRST inexact node — any inexactness in a child would already have raised
            # here — so it IS the introduction site, and its operands were all exact.
            # The EvalError is not an ArithmeticError, so it threads up untouched and
            # keeps this node's line, exactly like a child error (18.4).
            raise EvalError(_abort_message(self, result), line=self.line)
        return result

    def pretty(self) -> str:
        """Multi-line indented tree view, one node per line."""
        return "\n".join(self._pretty_lines(0))

    def _pretty_lines(self, depth: int) -> list[str]:
        label = "  " * depth + self._label()
        if self.value is not None:
            # 18.6/26: show what each PART got — value, type, scale, exactness, and
            # the mode's hex / exact-decimal / approximation details (Value.describe).
            label += f" Value = {self.value.describe()}"
        lines = [label]
        for child in self._children():
            lines.extend(child._pretty_lines(depth + 1))
        return lines


@dataclass(frozen=True, slots=True)
class Number(Node):
    """A numeric literal, kept exactly as written in the source (unsigned)."""

    lexeme: str
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.lexeme:
            raise ValueError("Number lexeme must not be empty")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.2/26.8: "LITERAL" with the source lexeme quoted, so it reads as the text.
        return f'LITERAL "{self.lexeme}"'

    def source(self) -> str:
        # The literal is kept verbatim, so its source IS the lexeme (35.2.2).
        return self.lexeme

    def _children(self) -> tuple[Node, ...]:
        return ()

    def _max_literal_scale(self) -> int:
        # The one node that carries a scale: read it off the verbatim lexeme (29.3).
        return _lexeme_scale(self.lexeme)

    def _evaluate(self, ctx: EvalContext) -> Value:
        # The mode only ever lands here, on the literals (18.3) — and so does the
        # min_fixed_point_precision floor (25.2.1), for the same reason; both ride
        # the context down and are unpacked at this leaf.
        return Value.from_lexeme(self.lexeme, ctx.mode, ctx.min_fixed_point_precision)


@dataclass(frozen=True, slots=True)
class VectorLiteral(Node):
    """A vector literal ``[a, b, …]`` — an ordered list of element sub-expressions.

    The one place a VECTOR Value is built (19.1.10). Strictly one-dimensional: each
    element evaluates in the run's chosen SCALAR mode, then ``Value.vector`` packs the
    results into a VECTOR carrying that element mode (and refuses a nested vector). The
    list may be EMPTY (``[]``). Like every node it has no mode of its own — the mode
    rides the context down to the element literals, exactly as for a function's args.
    """

    elements: tuple[Node, ...]
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.8: parallels LITERAL / CALL — VECTOR with its element count.
        return f"VECTOR ({len(self.elements)} elements)"

    def source(self) -> str:
        # The bracketed, comma-separated element sources (35.2.2); "[]" when empty.
        return "[" + ", ".join(element.source() for element in self.elements) + "]"

    def _operand_form(self, scale: int | None) -> str:
        # The bracket syntax over the elements' VALUES at the active precision (35.3.1).
        return "[" + ", ".join(_operand_value_string(e, scale) for e in self.elements) + "]"

    def _children(self) -> tuple[Node, ...]:
        return self.elements

    def _evaluate(self, ctx: EvalContext) -> Value:
        # Each element evaluates in the run's (scalar) mode; Value.vector packs them into
        # a one-dimensional VECTOR carrying that element mode (18.2 / 19.1.10).
        return Value.vector([element._walk(ctx) for element in self.elements], ctx.mode)


@dataclass(frozen=True, slots=True)
class UnaryOp(Node):
    """A unary sign operator applied to an operand."""

    op: str
    operand: Node
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.op not in UNARY_OPS:
            raise ValueError(f"unknown unary operator: {self.op!r}")
        _validate_line(self.line)

    def _label(self) -> str:
        return _UNARY_OPCODES[self.op]

    def source(self) -> str:
        # Prefix operator against its operand, e.g. -x or -(a + b) (35.2.2).
        return f"{self.op}{_parenthesize(self.operand)}"

    def _operand_form(self, scale: int | None) -> str:
        # The operator against the operand's VALUE, e.g. -3.00 (35.3.1).
        return f"{self.op}{_operand_value_string(self.operand, scale)}"

    def _children(self) -> tuple[Node, ...]:
        return (self.operand,)

    def _evaluate(self, ctx: EvalContext) -> Value:
        return _UNARY_FUNCS[self.op](self.operand._walk(ctx))


@dataclass(frozen=True, slots=True)
class BinOp(Node):
    """A binary operator; ``**`` is power and ``^`` is bitwise XOR (24.3.2)."""

    op: str
    left: Node
    right: Node
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.op not in BINARY_OPS:
            raise ValueError(f"unknown binary operator: {self.op!r}")
        _validate_line(self.line)

    def _label(self) -> str:
        return _BINARY_OPCODES[self.op]

    def source(self) -> str:
        # Infix operator with each operand parenthesized when it could re-associate
        # (35.2.2): "1 / 3", "a * (b + c)".
        return f"{_parenthesize(self.left)} {self.op} {_parenthesize(self.right)}"

    def _operand_form(self, scale: int | None) -> str:
        # Infix operator between the operands' VALUES at the active precision (35.3.1):
        # "1.00 / 3.00". Each operand is a single number, so no parenthesizing is needed.
        return (
            f"{_operand_value_string(self.left, scale)} {self.op} "
            f"{_operand_value_string(self.right, scale)}"
        )

    def _children(self) -> tuple[Node, ...]:
        return (self.left, self.right)

    def _evaluate(self, ctx: EvalContext) -> Value:
        # Plain arithmetic on Values — all type-faithful semantics live in Value (18.2).
        return _BINARY_FUNCS[self.op](
            self.left._walk(ctx),
            self.right._walk(ctx),
        )


@dataclass(frozen=True, slots=True)
class FuncCall(Node):
    """A function call like ``sqrt(x)`` or the nullary ``pi()`` (22.3 / 29.2): a
    NAME and its argument subtrees (empty for a nullary).

    Name and arity are validated in the parser (22.2 -> ParseError); the same
    checks here guard direct construction, mirroring UnaryOp/BinOp's op check.
    Evaluation forwards to the dispatched Value callable — an operand-method fed
    the evaluated arguments, or (for a nullary, 29.2) the run context — every
    per-mode semantic living there (18.2), so this node is mode-agnostic.
    """

    name: str
    args: tuple[Node, ...]
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.name not in _FUNCS
            and self.name not in _NULLARY_FUNCS
            and self.name not in _SPECIAL_FORM_NAMES
        ):
            raise ValueError(f"unknown function: {self.name!r}")
        if not _arity_ok(self.name, len(self.args)):
            lo, hi = FUNCTION_ARITIES[self.name]
            raise ValueError(f"{self.name!r} takes {_describe_arity(lo, hi)}, got {len(self.args)}")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.8: the call reads as CALL with the function name quoted, paralleling LITERAL.
        return f'CALL "{self.name}"'

    def source(self) -> str:
        # Call syntax with comma-separated arguments; a nullary renders as name() (35.2.2).
        return f"{self.name}({', '.join(arg.source() for arg in self.args)})"

    def _operand_form(self, scale: int | None) -> str:
        # Call syntax over the arguments' VALUES at the active precision (35.3.1):
        # "sqrt(2.00)"; a nullary like pi() renders with empty parentheses.
        if self.name in _SPECIAL_FORM_NAMES:
            # A special form (40.18) never walks its expr/var args as operands, so they
            # carry no value to lay out — fall back to the lexeme-level source rendering.
            return self.source()
        return f"{self.name}({', '.join(_operand_value_string(arg, scale) for arg in self.args)})"

    def _children(self) -> tuple[Node, ...]:
        return self.args

    def referenced_names(self) -> frozenset[str]:
        # A special form's named variable (40.18/40.19) is BOUND inside the call (a dummy
        # the form rebinds per sample/term), not a free reference — so drop it from the
        # names this subtree reads, keeping it from surfacing as a free unknown when the
        # form is nested in a solver expression. Which ARGUMENT holds the name differs per
        # form (_SPECIAL_FORM_BOUND_VAR). Every other function reads the base union.
        names = Node.referenced_names(self)
        if self.name in _SPECIAL_FORM_NAMES:
            var_node = self.args[_SPECIAL_FORM_BOUND_VAR[self.name]]
            if isinstance(var_node, Var):
                return names - {var_node.name}
        return names

    def _evaluate(self, ctx: EvalContext) -> Value:
        # A nullary takes the context, not operands (29.2); a special form (40.18) takes
        # the context and its UNEVALUATED argument subtrees; every other function is an
        # operand-method fed its evaluated arguments. Which path is settled by which
        # registry holds the name.
        if self.name in _SPECIAL_FORM_NAMES:
            # Deferred import (like reference.py's solver edge): the implementation lives
            # in the sibling engine expr.forms, which imports this module — a call-time
            # import keeps that edge out of module load and avoids a cycle.
            from mcp_abacus.expr.forms import dispatch_special_form

            return dispatch_special_form(self.name, ctx, self.args)
        if self.name in _NULLARY_FUNCS:
            return _NULLARY_FUNCS[self.name](ctx)
        operands = tuple(arg._walk(ctx) for arg in self.args)
        # Most functions take only scalars: refuse a vector operand here, the single
        # point every operand-function flows through, rather than a VECTOR arm in each
        # of the ~60 Value methods. The vector-CONSUMING functions (28.2/28.3 today;
        # covariance/correlation, 40.13/40.14, later) opt in via _VECTOR_FUNCS and
        # handle the VECTOR payload themselves (e.g. min/max reduce over its elements).
        if self.name not in _VECTOR_FUNCS and any(
            operand.mode is Mode.VECTOR for operand in operands
        ):
            raise NotRepresentableError(f"{self.name}() does not accept a vector")
        return _FUNCS[self.name](*operands)


@dataclass(frozen=True, slots=True)
class Assign(Node):
    """An assignment ``name = expr`` (30.3): a variable name and its value subtree.

    The loosest-precedence construct — it wraps a whole expression, never nests
    inside one (the parser only recognises it at statement level, 30.3). ``name``
    is the target variable's lexeme; ``expr`` is the right-hand expression whose
    Value the assignment binds. Evaluation walks ``expr``, stores the result under
    ``name`` in the run's VariableStore (30.2), and YIELDS that same Value, so an
    assignment also carries a value (the node's own computed ``value``) the way
    every other node does — and a later reference (30.4) reads it back.
    """

    name: str
    expr: Node
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Assign name must not be empty")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.8: ASSIGN with the target name quoted, paralleling LITERAL / CALL.
        return f'ASSIGN "{self.name}"'

    def source(self) -> str:
        # The binding form: target = right-hand side (35.2.2).
        return f"{self.name} = {self.expr.source()}"

    def _children(self) -> tuple[Node, ...]:
        return (self.expr,)

    def assigned_names(self) -> frozenset[str]:
        # The target name, plus any targets bound inside the RHS (folded via the base).
        # Calls Node.assigned_names explicitly: zero-arg super() misbehaves under the
        # slots=True dataclass rebuild (its __class__ cell points at the pre-slots class).
        return Node.assigned_names(self) | {self.name}

    def _evaluate(self, ctx: EvalContext) -> Value:
        # Bind the name to the RHS Value in the run store (30.2) and yield it; the
        # store lookup itself is the reference node's job (Var, 30.4).
        result = self.expr._walk(ctx)
        ctx.variables.set(self.name, result)
        return result


@dataclass(frozen=True, slots=True)
class Var(Node):
    """A variable reference — a bare NAME read back from the run's store (30.4).

    The counterpart to Assign (30.3): where Assign binds ``name = expr``, Var reads
    ``name``. The parser builds a Var only for a NAME NOT followed by ``(`` — a
    call or nullary keeps its FuncCall (22.2 / 29.2), so the two NAME forms stay
    distinct. Evaluation looks the name up in the run's VariableStore (30.2); an
    unset name raises UndefinedVariableError, which carries no position, so this
    node re-raises it as an EvalError tagged with its own source line (30.1).
    """

    name: str
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Var name must not be empty")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.8: VAR with the name quoted, paralleling LITERAL / CALL / ASSIGN.
        return f'VAR "{self.name}"'

    def source(self) -> str:
        # A bare reference is just the name (35.2.2).
        return self.name

    def _children(self) -> tuple[Node, ...]:
        return ()

    def referenced_names(self) -> frozenset[str]:
        return frozenset({self.name})

    def _evaluate(self, ctx: EvalContext) -> Value:
        try:
            return ctx.variables.get(self.name)
        except UndefinedVariableError as exc:
            # The store error has no line; position it here (30.1) so the run
            # surfaces an EvalError like every other evaluation failure.
            raise EvalError(str(exc), line=self.line) from exc


@dataclass(frozen=True, slots=True)
class Sequence(Node):
    """An ordered list of statements — a whole multi-statement program (30.5).

    Produced when the source holds more than one newline-separated statement; a
    single-statement input stays its bare node, with NO Sequence wrapper. Every
    statement is walked IN ORDER against the SAME run context, so an earlier
    ``x = ...`` assignment is visible to a later reference through the shared
    VariableStore (30.2). The Sequence YIELDS the LAST statement's Value — the
    program's result, the way a REPL echoes its final line — while the leading
    statements run for their effect (their bindings).
    """

    statements: tuple[Node, ...]
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.statements:
            raise ValueError("Sequence must hold at least one statement")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.8: SEQUENCE with the statement count, paralleling LITERAL / CALL.
        return f"SEQUENCE ({len(self.statements)})"

    def source(self) -> str:
        # The statements rejoined; "; " stands in for the newline separators (35.2.2).
        return "; ".join(statement.source() for statement in self.statements)

    def _children(self) -> tuple[Node, ...]:
        return self.statements

    def _evaluate(self, ctx: EvalContext) -> Value:
        # Walk every statement against the one ctx (shared store, 30.2); the leading
        # ones run for their bindings, the last one's Value is the program's result.
        *leading, last = self.statements
        for statement in leading:
            statement._walk(ctx)
        return last._walk(ctx)
