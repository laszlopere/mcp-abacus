# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the COMPLEX mode (19.1.8) — a + b*i over two fixed-point parts.

Complex is built on the FIXED_POINT engine, so the contract is: + - * stay EXACT
where the parts' arithmetic does, / and the transcendentals round half-to-even
onto the grid (inexact), and everything that needs a total order, the integers,
the bits, or a real-valued search REFUSES. The transcendentals are cross-checked
against the stdlib ``cmath`` ground truth at high precision.
"""

import asyncio
import cmath
import json
import math

import pytest

from mcp_abacus.expr import lexer
from mcp_abacus.expr.nodes import EvalError
from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import Complex, FixedPoint, Mode, NotRepresentableError, Value
from mcp_abacus.server import mcp


def _c(expr: str, floor: int = 12) -> Value:
    """Evaluate ``expr`` in complex mode at a fixed-point floor (decimals)."""
    return parse(expr).evaluate(Mode.COMPLEX, min_fixed_point_precision=floor)


def _as_complex(v: Value) -> complex:
    """The Value's two fixed-point parts as a Python complex, for cmath checks."""
    assert isinstance(v.payload, Complex)
    p = v.payload
    re = p.real.mantissa / 10**p.real.decimals
    im = p.imag.mantissa / 10**p.imag.decimals
    return complex(re, im)


# --- construction & literals ------------------------------------------------


def test_literal_real_and_imaginary():
    assert _as_complex(_c("3")) == 3 + 0j
    assert _as_complex(_c("4i")) == 0 + 4j
    assert _as_complex(_c("2.5i")) == 0 + 2.5j
    assert _as_complex(_c("3+4i")) == 3 + 4j
    assert _as_complex(_c("3-4i")) == 3 - 4j


def test_literal_j_suffix_is_accepted():
    assert _as_complex(_c("4j")) == 0 + 4j


def test_real_literal_is_exact():
    assert _c("3+4i").exact is True


def test_payload_validation_rejects_non_fixedpoint_parts():
    with pytest.raises(ValueError):
        Value(Mode.COMPLEX, "nope", exact=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Complex(FixedPoint(1, 0), 2)  # type: ignore[arg-type]


# --- arithmetic: exact landmarks --------------------------------------------


def test_mul_is_exact_on_the_grid():
    v = _c("(3+4i)*(1+2i)", floor=0)
    assert _as_complex(v) == -5 + 10j
    assert v.exact is True
    assert v.to_string() == "-5+10i"


def test_add_sub_exact():
    assert _c("(1+2i)+(3+4i)", floor=0).to_string() == "4+6i"
    assert _c("(1+2i)-(3+4i)", floor=0).to_string() == "-2-2i"


def test_i_squared_is_minus_one():
    v = _c("1i*1i", floor=0)
    assert _as_complex(v) == -1 + 0j
    assert v.exact is True


def test_division_rounds_like_fixed_point():
    # at scale 0 a half rounds to even (0); the faithful fixed-point behaviour.
    assert _c("1.0/(1+1i)").to_string() == "0.500000000000-0.500000000000i"
    assert _c("(1)/(1+1i)", floor=0).to_string() == "0"


def test_division_by_zero_raises():
    with pytest.raises(EvalError, match="zero"):
        _c("(1+1i)/(0+0i)")


def test_integer_power_is_exact():
    assert _c("(2+0i)**10", floor=0).to_string() == "1024"
    assert _c("(1+1i)**2", floor=0).to_string() == "2i"
    assert _c("(1+1i)**0", floor=0).to_string() == "1"
    assert _c("(2+0i)**-1").to_string() == "0.500000000000"


# --- abs / sqrt: exact where the real roots are -----------------------------


def test_abs_is_real_magnitude():
    v = _c("abs(3+4i)", floor=0)
    assert v.to_string() == "5"
    assert v.exact is True


def test_sqrt_principal_branch_exact_landmarks():
    assert _c("sqrt(-1)", floor=0).to_string() == "1i"
    assert _c("sqrt(3+4i)", floor=0).to_string() == "2+1i"


# --- conj / re / im / arg ----------------------------------------------------


def test_conj_re_im():
    assert _c("conj(3+4i)", floor=0).to_string() == "3-4i"
    assert _c("re(3+4i)", floor=0).to_string() == "3"
    assert _c("im(3+4i)", floor=0).to_string() == "4"


def test_arg_of_i_is_half_pi():
    assert _as_complex(_c("arg(1i)", floor=12)).real == pytest.approx(math.pi / 2, abs=1e-10)


def test_re_im_conj_on_real_modes():
    assert parse("conj(5)").evaluate(Mode.FIXED_POINT).to_string() == "5"
    assert parse("re(5)").evaluate(Mode.RATIONAL).to_string() == "5"
    assert parse("im(5)").evaluate(Mode.FIXED_POINT).to_string() == "0"
    # arg of a positive real is exactly 0 (atan2's exact landmark)
    assert parse("arg(5)").evaluate(Mode.FIXED_POINT).to_string() == "0"


# --- transcendentals cross-checked against cmath ----------------------------

_CMATH_CASES = [
    ("exp(1+1i)", lambda: cmath.exp(1 + 1j)),
    ("exp(1i*pi)", lambda: cmath.exp(1j * math.pi)),
    ("ln(1i)", lambda: cmath.log(1j)),
    ("ln(3+4i)", lambda: cmath.log(3 + 4j)),
    ("sqrt(1+2i)", lambda: cmath.sqrt(1 + 2j)),
    ("sin(1+1i)", lambda: cmath.sin(1 + 1j)),
    ("cos(1+1i)", lambda: cmath.cos(1 + 1j)),
    ("tan(1+1i)", lambda: cmath.tan(1 + 1j)),
    ("sinh(1+1i)", lambda: cmath.sinh(1 + 1j)),
    ("cosh(1+1i)", lambda: cmath.cosh(1 + 1j)),
    ("tanh(1+1i)", lambda: cmath.tanh(1 + 1j)),
    ("(1+2i)**(0.5)", lambda: (1 + 2j) ** 0.5),
    ("(2+1i)**(1+1i)", lambda: (2 + 1j) ** (1 + 1j)),
    ("log10(100)", lambda: cmath.log10(100)),
    ("log2(8+0i)", lambda: cmath.log(8) / cmath.log(2)),
]


@pytest.mark.parametrize("expr,expected", _CMATH_CASES, ids=[c[0] for c in _CMATH_CASES])
def test_transcendentals_match_cmath(expr, expected):
    got = _as_complex(_c(expr, floor=12))
    want = expected()
    assert got.real == pytest.approx(want.real, abs=1e-9)
    assert got.imag == pytest.approx(want.imag, abs=1e-9)


# --- formatting --------------------------------------------------------------


def _cv(re_mant: int, im_mant: int, scale: int = 0) -> str:
    payload = Complex(FixedPoint(re_mant, scale), FixedPoint(im_mant, scale))
    return Value(Mode.COMPLEX, payload, exact=True).to_string()


def test_to_string_variants():
    assert _cv(0, 0) == "0"
    assert _cv(5, 0) == "5"
    assert _cv(0, 1) == "1i"
    assert _cv(0, -1) == "-1i"
    assert _cv(3, -4) == "3-4i"


def test_precision_is_the_wider_part_scale():
    v = Value(Mode.COMPLEX, Complex(FixedPoint(10, 1), FixedPoint(200, 2)), exact=True)
    assert v.precision() == 2


# --- refusals: ordering, bits, integers, rounding, real-only ----------------

_REFUSED = [
    "min(1+1i, 2)",
    "max(1+1i, 2)",
    "median(1+1i, 2, 3)",
    "clamp(1+1i, 0, 2)",
    "sign(1+1i)",
    "floor(1.5+2.5i)",
    "ceil(1.5+2.5i)",
    "round(1.5+2.5i)",
    "trunc(1.5+2.5i)",
    "(3+4i)//2",
    "(3+4i)%2",
    "(1+1i)&2",
    "(1+1i)|2",
    "(1+1i)^2",  # ^ is XOR, not power
    "~(1+1i)",
    "gcd(4+0i, 6)",
    "lcm(4+0i, 6)",
    "factorial(3+0i)",
    "comb(5+0i, 2)",
    "perm(5+0i, 2)",
    "cbrt(8+0i)",
    "cot(1+1i)",
    "asin(0.5+0i)",
    "acos(0.5+0i)",
    "atan(1+1i)",
    "atan2(1+0i, 1)",
    "degrees(1+1i)",
    "radians(90+0i)",
    "asinh(1+1i)",
    "acosh(2+0i)",
    "atanh(0.5+0i)",
    "time()",
]


@pytest.mark.parametrize("expr", _REFUSED)
def test_complex_refuses(expr):
    # the node walk wraps the operand-method's domain error in an EvalError (it
    # attaches the source line); the underlying cause is a NotRepresentableError.
    with pytest.raises(EvalError) as exc:
        parse(expr).evaluate(Mode.COMPLEX, min_fixed_point_precision=4)
    assert "complex" in str(exc.value).lower() or "ordering" in str(exc.value).lower()


def test_imaginary_literal_refused_outside_complex_mode():
    for mode in (Mode.FIXED_POINT, Mode.FLOATING_POINT, Mode.RATIONAL):
        with pytest.raises(NotRepresentableError):
            Value.from_lexeme("4i", mode)


# --- lexer: the imaginary suffix --------------------------------------------


def test_lexer_imaginary_suffix_is_one_number_token():
    for src in ("4i", "2.5i", "1e3i", "4j"):
        toks = lexer.tokenize(src)
        assert toks[0].kind == lexer.NUMBER and toks[0].lexeme == src
        assert toks[1].kind == lexer.EOF


def test_lexer_bare_i_is_a_name():
    toks = lexer.tokenize("i")
    assert toks[0].kind == lexer.NAME and toks[0].lexeme == "i"
    # `i` stays usable as a variable
    assert parse("i = 5\ni*1i").evaluate(Mode.COMPLEX).to_string() == "5i"


def test_lexer_rejects_imaginary_on_hex():
    with pytest.raises(lexer.LexError):
        lexer.tokenize("0x1fi")


# --- end-to-end through the calculate / solver tools ------------------------


def _call(tool, **args):
    res = asyncio.run(mcp.call_tool(tool, args))
    blocks = res[0] if isinstance(res, tuple) else res
    return json.loads(blocks[0].text)


def test_calculate_complex_reports_exact():
    p = _call("calculate", expression="(3+4i)*(1+2i)", mode="complex")
    assert p["error"] is None
    assert p["values"][0]["value"] == "-5+10i (exact)"


def test_calculate_complex_inexact_verdict():
    p = _call("calculate", expression="exp(1i*pi)", mode="complex", min_fixed_point_precision=6)
    assert p["error"] is None
    assert "inexact" in p["values"][0]["value"]


def test_solver_refuses_complex():
    p = _call("solver", expression="x-1", variable="x", lower=0, upper=2, mode="complex")
    assert p["solution"] is None
    assert "complex" in p["error"].lower()
