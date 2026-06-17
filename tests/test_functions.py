# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Functional tests for the abacus call functions (TODO 22).

Each test exercises ONE function — abs or sqrt — through the in-process
`calculate` tool across the three modes and a spread of arguments, asserting the
tool's annotated `value` string (e.g. "16 (exact)") and, for the domain refusals,
the line-tagged error. Driving the real tool (full dispatch -> parse -> evaluate
-> render) keeps this a functional test, not a unit test of the methods.

Under `pytest -v` (run-tests.sh --verbose) conftest's _compact_functions_trace
prints every "expression [mode] = value" pair it sees on the tool seam.
"""

import asyncio
import json
from fractions import Fraction

import pytest

from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import FixedPoint, Mode
from mcp_abacus.server import mcp


def _calc(expression, mode=None, floor=None):
    """Invoke `calculate` in-process; return its structured payload dict (25.1).

    `mode` is omitted from the request when None, exercising the fixed-point default.
    `floor` (min_fixed_point_precision) is likewise omitted when None — supplied only
    by the nullary-precision tests (29.4) that raise the derived scale past the
    literals' own.
    """
    arguments = {"expression": expression}
    if mode is not None:
        arguments["mode"] = mode
    if floor is not None:
        arguments["min_fixed_point_precision"] = floor
    result = asyncio.run(mcp.call_tool("calculate", arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def _value(expression, mode=None, floor=None):
    """The annotated `value` string on success; fail loudly if the call errored."""
    payload = _calc(expression, mode, floor)
    assert payload["error"] is None, f"{expression!r} [{mode}] errored: {payload['error']}"
    return payload["value"]


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # abs is exact wherever the mode has an exact grid — magnitude only drops a
        # sign. Negatives reach abs through unary minus or a subexpression.
        ("abs(-3)", None, "3 (exact)"),  # fixed-point default
        ("abs(3)", None, "3 (exact)"),
        ("abs(2 - 5)", None, "3 (exact)"),  # negative via a subexpression
        ("abs(-2.50)", None, "2.50 (exact)"),  # scale preserved
        ("abs(0)", None, "0 (exact)"),
        ("abs(-3)", "rational", "3 (exact)"),
        ("abs(-1/3)", "rational", "1/3 (exact)"),
        # binary64 carries its inexact flag through abs (the literal is already inexact).
        ("abs(-3)", "floating-point", "3.0 (inexact)"),
        ("abs(-2.5)", "floating-point", "2.5 (inexact)"),
    ],
)
def test_abs(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # sum is variadic and behaves like repeated +: exact in every mode, with the
        # fixed-point result at the covering (max) scale of its operands.
        ("sum(1, 2, 3, 4)", None, "10 (exact)"),  # fixed-point default
        ("sum(5)", None, "5 (exact)"),  # single arg -> identity (the empty fold)
        ("sum(1.5, 2.25)", None, "3.75 (exact)"),  # covering scale 2, no rounding
        ("sum(1/2, 1/3, 1/6)", "rational", "1 (exact)"),  # 3/6 + 2/6 + 1/6 = 1
        ("sum(1/3)", "rational", "1/3 (exact)"),
        # binary64 carries its operands' inexact flag through (like every float op).
        ("sum(1, 2, 3)", "floating-point", "6.0 (inexact)"),
    ],
)
def test_sum(expression, mode, value):
    assert _value(expression, mode) == value


def test_sum_inherits_operand_inexactness():
    # An inexact operand (fixed-point sqrt rounds) makes the whole sum inexact,
    # exactly as repeated + would — the fold propagates the flag.
    assert _value("sum(1, sqrt(2))").startswith("2 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # max/min SELECT an operand — never compute — so the result is that operand
        # verbatim: its own scale, no covering, no rounding.
        ("max(1, 2, 3)", None, "3 (exact)"),  # fixed-point default
        ("min(1, 2, 3)", None, "1 (exact)"),
        ("max(5)", None, "5 (exact)"),  # single arg -> identity
        ("min(5)", None, "5 (exact)"),
        ("max(-3, -5)", None, "-3 (exact)"),  # ordering respects sign
        ("min(-3, -5)", None, "-5 (exact)"),
        # The chosen operand keeps its OWN scale — no covering to a common one.
        ("max(1.5, 2.250)", None, "2.250 (exact)"),  # scale 3 preserved
        ("min(1.5, 2.250)", None, "1.5 (exact)"),  # scale 1 preserved
        # A tie keeps the EARLIER operand (so its scale wins) — stable selection.
        ("max(1.5, 1.50)", None, "1.5 (exact)"),
        ("min(1.50, 1.5)", None, "1.50 (exact)"),
        # rational compares exact fractions.
        ("max(1/2, 2/3, 1/6)", "rational", "2/3 (exact)"),
        ("min(1/2, 2/3, 1/6)", "rational", "1/6 (exact)"),
        # binary64 selects too, carrying the operand's inexact flag.
        ("max(1, 2, 3)", "floating-point", "3.0 (inexact)"),
        ("min(1.5, 2.5)", "floating-point", "1.5 (inexact)"),
    ],
)
def test_max_min(expression, mode, value):
    assert _value(expression, mode) == value


def test_max_min_carry_the_chosen_operands_exactness():
    # Selection carries the picked operand's own exactness: max keeps the exact 2,
    # min picks the inexact sqrt(2) (which rounds in fixed-point) and is inexact.
    assert _value("max(2, sqrt(2))") == "2 (exact)"
    assert _value("min(2, sqrt(2))").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # product is variadic and behaves like repeated *: unlike sum it COMPUTES, so
        # fixed-point may round to the covering scale, rational stays exact, float rounds.
        ("product(2, 3, 4)", None, "24 (exact)"),  # fixed-point default
        ("product(5)", None, "5 (exact)"),  # single arg -> identity (the empty fold)
        ("product(1.5, 2.0)", None, "3.0 (exact)"),  # covering scale 1, fits exactly
        # 1.5 * 1.5 = 2.25 but the covering scale is 1 -> rounds, flagged inexact.
        (
            "product(1.5, 1.5)",
            None,
            "2.2 (inexact, rounded to 1 decimal — pass min_fixed_point_precision "
            "for more; e.g. =5 → 2.25000)",
        ),
        ("product(1/2, 2/3, 3/4)", "rational", "1/4 (exact)"),  # exact fractions
        # binary64 carries its operands' inexact flag through (like every float op).
        ("product(2, 3)", "floating-point", "6.0 (inexact)"),
    ],
)
def test_product(expression, mode, value):
    assert _value(expression, mode) == value


def test_product_inherits_operand_inexactness():
    # An inexact operand (fixed-point sqrt rounds) makes the whole product inexact,
    # exactly as repeated * would — the fold propagates the flag.
    assert _value("product(1, sqrt(2))").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # avg COMPUTES (sum / count), so it follows the mode's / rule: fixed-point
        # quantizes to the covering scale, rational is exact, float rounds.
        ("avg(2, 4, 6)", None, "4 (exact)"),  # fixed-point default, divides evenly
        # (1 + 2) / 2 = 1.5 but the covering scale is 0 -> rounds, flagged inexact.
        (
            "avg(1, 2)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5000)",
        ),
        ("avg(5)", None, "5 (exact)"),  # single arg -> a / 1, the identity
        ("avg(1.0, 2.0)", None, "1.5 (exact)"),  # scale 1 covers the halving
        ("avg(1, 2, 3)", "rational", "2 (exact)"),  # rational divides exactly
        ("avg(1, 2)", "rational", "3/2 (exact)"),  # ... even when it does not reduce
        # binary64 rounds and carries the inexact flag.
        ("avg(2, 4)", "floating-point", "3.0 (inexact)"),
    ],
)
def test_avg(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # median orders by value (no arithmetic) and, for an ODD count, SELECTS the
        # middle operand verbatim — exact, carrying its own scale, like max/min.
        ("median(3, 1, 2)", None, "2 (exact)"),  # fixed-point default, unsorted input
        ("median(5)", None, "5 (exact)"),  # single arg -> identity
        ("median(-5, -1, -3)", None, "-3 (exact)"),  # ordering respects sign
        ("median(1/2, 1/6, 2/3)", "rational", "1/2 (exact)"),  # exact fractions
        ("median(1, 2, 3)", "floating-point", "2.0 (inexact)"),  # selects, carries flag
        # An EVEN count averages the two middles -> follows the / rule, like avg.
        ("median(1, 3)", None, "2 (exact)"),  # (1 + 3) / 2 = 2 fits scale 0 exactly
        # (1 + 2) / 2 of the two middles of 1,2,3,4 = 1.5 -> rounds at scale 0.
        (
            "median(1, 2, 3, 4)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.5000)",
        ),
        ("median(2, 4)", "rational", "3 (exact)"),  # even average, exact in rational
    ],
)
def test_median(expression, mode, value):
    assert _value(expression, mode) == value


def test_median_odd_carries_the_selected_operands_exactness():
    # An odd-count median SELECTS the middle operand verbatim, so it carries that
    # operand's own exactness — here the inexact (rounded) fixed-point sqrt(2).
    assert _value("median(1, sqrt(2), 3)").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # POPULATION variance (/ n): sum of squared deviations from the mean / n.
        # The textbook 2,4,4,4,5,5,7,9 has mean 5, squared-deviation sum 32, /8 = 4 —
        # all-integer arithmetic, so exact in every mode.
        ("variance(2, 4, 4, 4, 5, 5, 7, 9)", None, "4 (exact)"),  # fixed-point default
        ("variance(2, 4, 4, 4, 5, 5, 7, 9)", "rational", "4 (exact)"),
        ("variance(2, 4, 4, 4, 5, 5, 7, 9)", "floating-point", "4.0 (inexact)"),
        ("variance(1, 2, 3, 4, 5)", None, "2 (exact)"),  # mean 3, sum 10, /5 = 2
        ("variance(5)", None, "0 (exact)"),  # a lone point has no spread
        # rational keeps the exact mean and divide even when they do not reduce.
        ("variance(1, 2)", "rational", "1/4 (exact)"),  # mean 3/2, devs ±1/2, /2
        ("variance(2, 4, 6)", "rational", "8/3 (exact)"),  # mean 4, sum 8, /3
        # binary64 rounds; fixed-point at scale 0 rounds 8/3 and flags inexact.
        ("variance(2, 4, 6)", "floating-point", "2.6666666666666665 (inexact)"),
        (
            "variance(2, 4, 6)",
            None,
            "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.6667)",
        ),
    ],
)
def test_variance(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # stddev == sqrt(variance), inheriting sqrt's per-mode story. The textbook
        # set's variance 4 has an exact root 2 in every mode (a perfect square).
        ("stddev(2, 4, 4, 4, 5, 5, 7, 9)", None, "2 (exact)"),  # fixed-point default
        ("stddev(2, 4, 4, 4, 5, 5, 7, 9)", "rational", "2 (exact)"),
        ("stddev(2, 4, 4, 4, 5, 5, 7, 9)", "floating-point", "2.0 (inexact)"),
        ("stddev(1, 2)", "rational", "1/2 (exact)"),  # variance 1/4, exact root
        ("stddev(5)", None, "0 (exact)"),  # sqrt(0) = 0
        # Irrational root: float/fixed-point round and flag inexact (rational refuses,
        # tested separately). variance(1,2,3,4,5) = 2, sqrt(2) is irrational.
        ("stddev(1, 2, 3, 4, 5)", "floating-point", "1.4142135623730951 (inexact)"),
        (
            "stddev(1, 2, 3, 4, 5)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.4142)",
        ),
    ],
)
def test_stddev(expression, mode, value):
    assert _value(expression, mode) == value


def test_stddev_rational_refuses_an_irrational_root():
    # In rational mode stddev inherits sqrt's exact-or-refuse pitch: variance 2 has
    # no rational root, so it raises rather than fabricate digits (line-tagged).
    payload = _calc("stddev(1, 2, 3, 4, 5)", "rational")
    assert payload["error"] == "rational square root is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Fixed-point: exact only when the root lands on the grid (a perfect square at
        # the operand's scale); otherwise it rounds to that scale and flags inexact,
        # offering more decimals via min_fixed_point_precision.
        ("sqrt(16)", None, "4 (exact)"),
        ("sqrt(2.25)", None, "1.50 (exact)"),  # 1.5 is exact at scale 2
        (
            "sqrt(2)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.4142)",
        ),
        (
            "sqrt(2.000000)",
            None,
            "1.414214 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.4142135624)",
        ),
        # binary64 sqrt is unconditionally inexact, even for a perfect square.
        ("sqrt(16)", "floating-point", "4.0 (inexact)"),
        ("sqrt(2)", "floating-point", "1.4142135623730951 (inexact)"),
        # Rational is exact only for a perfect square (both parts square).
        ("sqrt(16)", "rational", "4 (exact)"),
        ("sqrt(2.25)", "rational", "3/2 (exact)"),  # 9/4 -> 3/2
    ],
)
def test_sqrt(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # No real root for a negative operand in any mode...
        ("sqrt(-4)", None, "square root of a negative value"),
        ("sqrt(-4)", "floating-point", "square root of a negative value"),
        ("sqrt(-1)", "rational", "square root of a negative value"),
        # ...and rational refuses an irrational root (no scale to round to).
        ("sqrt(2)", "rational", "rational square root is irrational"),
    ],
)
def test_sqrt_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # The BINARY pow(x, y) function — the call form of ** (28.20). Integer
        # exponent is exact in the exact modes, float rounds.
        ("pow(2, 10)", None, "1024 (exact)"),
        ("pow(2, 10)", "rational", "1024 (exact)"),
        ("pow(2, 10)", "floating-point", "1024.0 (inexact)"),
        # FRACTIONAL fixed-point exponent (28.20.1). PATH A — a perfect q-th root is
        # exact (result scale covers base and exponent).
        ("pow(4, 0.5)", None, "2.0 (exact)"),  # sqrt(4)
        ("pow(0.25, 0.5)", None, "0.50 (exact)"),  # sqrt(1/4)
        ("pow(32, 0.2)", None, "2.0 (exact)"),  # fifth root of 32
        # PATH B — an irrational root computes via exp(y*ln x), inexact.
        (
            "pow(2.000000, 0.5)",
            None,
            "1.414214 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.4142135624)",
        ),
        ("pow(2, 0.5)", "floating-point", "1.4142135623730951 (inexact)"),
    ],
)
def test_pow(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # A non-integer exponent is irrational for a rational base (no scale).
        ("pow(2, 0.5)", "rational", "rational power requires an integer exponent"),
        # An even root of a negative base is complex — no real value.
        ("pow(-4, 0.5)", None, "even root of a negative value"),
        # An odd irrational root of a negative base cannot go through ln — refuse.
        ("pow(-2, 0.2)", None, "fractional power of a negative base is irrational"),
        # Zero to a negative power divides by zero.
        ("pow(0, -2)", None, "fixed-point zero to a negative power"),
    ],
)
def test_pow_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # sin(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("sin(0)", None, "0 (exact)"),
        ("sin(0)", "rational", "0 (exact)"),
        ("sin(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.sin and is unconditionally inexact.
        ("sin(1)", "floating-point", "0.8414709848078965 (inexact)"),
        ("sin(0.5)", "floating-point", "0.479425538604203 (inexact)"),
        # Fixed-point sums a Taylor series at the operand's scale and flags inexact;
        # the result rounds to that scale (sin(1) ~ 0.8415 -> 1 at scale 0).
        (
            "sin(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.8415)",
        ),
        # Inexact fixed-point results also carry the min_fixed_point_precision offer
        # hint (the worked example at scale+4), exactly like sqrt(2.000000).
        (
            "sin(1.000000)",  # math.sin(1) at scale 6
            None,
            "0.841471 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.8414709848)",
        ),
        (
            "sin(0.500000)",
            None,
            "0.479426 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.4794255386)",
        ),
        # Negative argument (sin is odd) — exercises the sign-fold.
        (
            "sin(-1.500000)",
            None,
            "-0.997495 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.9974949866)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "sin(10.000000)",  # math.sin(10)
            None,
            "-0.544021 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.5440211109)",
        ),
        (
            "sin(1000.000000)",  # math.sin(1000), heavily range-reduced
            None,
            "0.826880 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.8268795405)",
        ),
    ],
)
def test_sin(expression, mode, value):
    assert _value(expression, mode) == value


def test_sin_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("sin(1)", "rational")
    assert payload["error"] == "sine of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # cos(0) = 1 is the one exact case in every mode (transcendental otherwise).
        ("cos(0)", None, "1 (exact)"),
        ("cos(0)", "rational", "1 (exact)"),
        ("cos(0)", "floating-point", "1.0 (inexact)"),
        # binary64 uses math.cos and is unconditionally inexact.
        ("cos(1)", "floating-point", "0.5403023058681398 (inexact)"),
        ("cos(0.5)", "floating-point", "0.8775825618903728 (inexact)"),
        # Fixed-point sums the even Taylor series at the operand's scale, inexact;
        # cos(1) ~ 0.5403 rounds to 1 at scale 0 (nearest whole number).
        (
            "cos(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.5403)",
        ),
        (
            "cos(1.000000)",  # math.cos(1) at scale 6
            None,
            "0.540302 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.5403023059)",
        ),
        (
            "cos(0.500000)",
            None,
            "0.877583 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.8775825619)",
        ),
        # cos is even — a negative argument gives the same result as its magnitude.
        (
            "cos(-2.000000)",
            None,
            "-0.416147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.4161468365)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "cos(10.000000)",  # math.cos(10)
            None,
            "-0.839072 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.8390715291)",
        ),
    ],
)
def test_cos(expression, mode, value):
    assert _value(expression, mode) == value


def test_cos_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("cos(1)", "rational")
    assert payload["error"] == "cosine of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # tan(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("tan(0)", None, "0 (exact)"),
        ("tan(0)", "rational", "0 (exact)"),
        ("tan(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.tan and is unconditionally inexact.
        ("tan(1)", "floating-point", "1.5574077246549023 (inexact)"),
        ("tan(0.5)", "floating-point", "0.5463024898437905 (inexact)"),
        # Fixed-point divides the sine and cosine series at the working scale, inexact;
        # tan(1) ~ 1.5574 rounds to 2 at scale 0 (nearest whole number).
        (
            "tan(1)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5574)",
        ),
        (
            "tan(1.000000)",  # math.tan(1) at scale 6
            None,
            "1.557408 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5574077247)",
        ),
        (
            "tan(0.500000)",
            None,
            "0.546302 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.5463024898)",
        ),
        # Negative argument (tan is odd) — exercises the sine sign-fold.
        (
            "tan(-1.000000)",
            None,
            "-1.557408 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -1.5574077247)",
        ),
        # Second quadrant: cos is negative, so tan comes out negative.
        (
            "tan(2.000000)",  # math.tan(2)
            None,
            "-2.185040 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -2.1850398633)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "tan(10.000000)",  # math.tan(10)
            None,
            "0.648361 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6483608275)",
        ),
        # Near an odd multiple of pi/2 the cosine is tiny but never exactly 0 (pi is
        # irrational), so the answer is just a large inexact value, not an error.
        (
            "tan(1.570796)",  # math.tan(1.570796), just shy of pi/2
            None,
            "3060023.306194 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 3060023.3061935647)",
        ),
    ],
)
def test_tan(expression, mode, value):
    assert _value(expression, mode) == value


def test_tan_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("tan(1)", "rational")
    assert payload["error"] == "tangent of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # cot == cos/sin, the mirror of tan: transcendental, so inexact everywhere it
        # is defined — and UNLIKE tan there is no exact case, since cot(0) is undefined
        # (sin = 0). binary64 uses math.cos / math.sin and is unconditionally inexact.
        ("cot(1)", "floating-point", "0.6420926159343308 (inexact)"),
        ("cot(0.5)", "floating-point", "1.830487721712452 (inexact)"),
        # Fixed-point divides the cosine and sine series at the working scale, inexact;
        # cot(1) ~ 0.6421 rounds to 1 at scale 0 (nearest whole number).
        (
            "cot(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.6421)",
        ),
        (
            "cot(1.000000)",  # cos/sin of 1 at scale 6
            None,
            "0.642093 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6420926159)",
        ),
        (
            "cot(0.500000)",
            None,
            "1.830488 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.8304877217)",
        ),
        # Negative argument (cot is odd) — exercises the sine sign-fold.
        (
            "cot(-1.000000)",
            None,
            "-0.642093 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.6420926159)",
        ),
        # Second quadrant: cos is negative, so cot comes out negative.
        (
            "cot(2.000000)",  # cos/sin of 2
            None,
            "-0.457658 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.4576575544)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "cot(10.000000)",  # cos/sin of 10
            None,
            "1.542351 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5423510454)",
        ),
        # Near a multiple of pi the sine is tiny but never exactly 0 (pi is
        # irrational), so the answer is just a large inexact value, not an error.
        (
            "cot(3.141592)",  # cos/sin of 3.141592, just shy of pi
            None,
            "-1530011.653097 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -1530011.6530966189)",
        ),
    ],
)
def test_cot(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        # cot(0) = cos/sin = 1/0 is undefined in every mode (the mirror of tan(0) = 0,
        # which IS defined): sin = 0 there, so each mode raises its division/domain error.
        (None, "fixed-point cotangent of a multiple of pi"),
        ("rational", "cotangent of zero is undefined"),
        ("floating-point", "float division by zero"),
    ],
)
def test_cot_of_zero_is_undefined_with_a_line_tagged_error(mode, error):
    payload = _calc("cot(0)", mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_cot_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("cot(1)", "rational")
    assert payload["error"] == "cotangent of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # asin(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("asin(0)", None, "0 (exact)"),
        ("asin(0)", "rational", "0 (exact)"),
        ("asin(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.asin and is unconditionally inexact; asin(1) = pi/2.
        ("asin(1)", "floating-point", "1.5707963267948966 (inexact)"),
        ("asin(0.5)", "floating-point", "0.5235987755982989 (inexact)"),
        # asin is odd — a negative argument negates the result (the sign-fold).
        ("asin(-0.5)", "floating-point", "-0.5235987755982989 (inexact)"),
        # Fixed-point routes through the arctan series and flags inexact; asin(1) =
        # pi/2 ~ 1.5708 rounds to 2 at scale 0 (nearest whole number).
        (
            "asin(1)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5708)",
        ),
        # asin(1/2) = pi/6; the inexact result carries the precision-offer hint.
        (
            "asin(0.500000)",
            None,
            "0.523599 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.5235987756)",
        ),
        # Negative argument (asin is odd) — exercises the fixed-point sign-fold.
        (
            "asin(-0.500000)",
            None,
            "-0.523599 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.5235987756)",
        ),
        # |x| > 1/sqrt(2): the arctan argument exceeds 1, so asin reduces via
        # pi/2 - atan(1/u). asin(sqrt(3)/2) ~ pi/3 (0.866025 is just shy of it).
        (
            "asin(0.866025)",
            None,
            "1.047197 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.0471967436)",
        ),
        # The domain endpoint asin(1) = pi/2 at a non-zero scale (root = 0 path).
        (
            "asin(1.000000)",
            None,
            "1.570796 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5707963268)",
        ),
    ],
)
def test_asin(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    "mode",
    [None, "floating-point", "rational"],
)
def test_asin_refuses_outside_the_domain_with_a_line_tagged_error(mode):
    # |x| > 1 has no real arcsine: every mode refuses with the same domain message.
    payload = _calc("asin(2)", mode)
    assert payload["error"] == "arcsine argument outside the domain [-1, 1]"
    assert payload["value"] is None


def test_asin_refuses_a_non_zero_rational_with_a_line_tagged_error():
    # In-domain but irrational: a non-zero rational arcsine refuses (exact-or-refuse).
    payload = _calc("asin(0.5)", "rational")
    assert payload["error"] == "arcsine of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # acos(1) = 0 is the one exact case in every mode (acos = pi/2 - asin, and
        # asin(1) = pi/2 cancels). NB the exact landmark is x = 1, not x = 0.
        ("acos(1)", None, "0 (exact)"),
        ("acos(1)", "rational", "0 (exact)"),
        ("acos(1)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.acos and is unconditionally inexact; acos(0) = pi/2.
        ("acos(0)", "floating-point", "1.5707963267948966 (inexact)"),
        ("acos(0.5)", "floating-point", "1.0471975511965979 (inexact)"),
        # acos is NOT odd: acos(-x) = pi - acos(x), so a negative argument lands in
        # (pi/2, pi]. acos(-1) = pi is the far endpoint.
        ("acos(-0.5)", "floating-point", "2.0943951023931957 (inexact)"),
        ("acos(-1)", "floating-point", "3.141592653589793 (inexact)"),
        # Fixed-point subtracts asin from pi/2 and flags inexact; acos(0) = pi/2 ~
        # 1.5708 rounds to 2 at scale 0 (acos(0) is irrational, unlike asin(0) = 0).
        (
            "acos(0)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5708)",
        ),
        # acos(1/2) = pi/3; the inexact result carries the precision-offer hint.
        (
            "acos(0.500000)",
            None,
            "1.047198 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.0471975512)",
        ),
        # Negative argument (acos(-1/2) = 2*pi/3) — exercises the sign handling.
        (
            "acos(-0.500000)",
            None,
            "2.094395 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 2.0943951024)",
        ),
        # acos(-1) = pi, the far domain endpoint.
        (
            "acos(-1.000000)",
            None,
            "3.141593 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 3.1415926536)",
        ),
        # acos(1) = 0 stays EXACT at a non-zero scale (the mantissa == 10**decimals path).
        ("acos(1.000000)", None, "0.000000 (exact)"),
    ],
)
def test_acos(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    "mode",
    [None, "floating-point", "rational"],
)
def test_acos_refuses_outside_the_domain_with_a_line_tagged_error(mode):
    # |x| > 1 has no real arccosine: every mode refuses with the same domain message.
    payload = _calc("acos(2)", mode)
    assert payload["error"] == "arccosine argument outside the domain [-1, 1]"
    assert payload["value"] is None


def test_acos_refuses_a_rational_other_than_one_with_a_line_tagged_error():
    # In-domain but irrational: any rational != 1 arccosine refuses (exact-or-refuse).
    payload = _calc("acos(0.5)", "rational")
    assert payload["error"] == "arccosine of a rational other than 1 is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # atan(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("atan(0)", None, "0 (exact)"),
        ("atan(0)", "rational", "0 (exact)"),
        ("atan(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.atan and is unconditionally inexact.
        ("atan(0.5)", "floating-point", "0.4636476090008061 (inexact)"),
        ("atan(1)", "floating-point", "0.7853981633974483 (inexact)"),  # pi/4
        ("atan(2)", "floating-point", "1.1071487177940904 (inexact)"),  # over-1 argument
        # atan is odd: atan(-x) = -atan(x).
        ("atan(-0.5)", "floating-point", "-0.4636476090008061 (inexact)"),
        ("atan(-2)", "floating-point", "-1.1071487177940904 (inexact)"),
        # No domain limit — a large argument just approaches pi/2 from below.
        ("atan(1000)", "floating-point", "1.5697963271282298 (inexact)"),
        # Fixed-point sums the arctan series and flags inexact; atan(1) = pi/4 ~ 0.7854
        # rounds to 1 at scale 0 (atan(1) is irrational, the only exact case is x = 0).
        (
            "atan(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.7854)",
        ),
        # atan(1/2); the inexact result carries the precision-offer hint.
        (
            "atan(0.500000)",
            None,
            "0.463648 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.4636476090)",
        ),
        # An over-1 argument exercises the atan(x) = pi/2 - atan(1/x) reduction.
        (
            "atan(2.000000)",
            None,
            "1.107149 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.1071487178)",
        ),
        # Negative argument (atan is odd) — exercises the sign handling.
        (
            "atan(-0.500000)",
            None,
            "-0.463648 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.4636476090)",
        ),
        (
            "atan(-2.000000)",
            None,
            "-1.107149 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -1.1071487178)",
        ),
        # A large argument nears pi/2 ~ 1.5708 without ever reaching it (no domain cap).
        (
            "atan(1000.000000)",
            None,
            "1.569796 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5697963271)",
        ),
        # atan(0) = 0 stays EXACT at a non-zero scale (the mantissa == 0 path).
        ("atan(0.000000)", None, "0.000000 (exact)"),
    ],
)
def test_atan(expression, mode, value):
    assert _value(expression, mode) == value


def test_atan_refuses_a_non_zero_rational_with_a_line_tagged_error():
    # atan of a rational is irrational except atan(0) = 0; any non-zero rational refuses
    # (exact-or-refuse). NB no domain refusal — atan accepts every real.
    payload = _calc("atan(2)", "rational")
    assert payload["error"] == "arctangent of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # log(1) = 0 is the one exact case in every mode (transcendental otherwise);
        # `ln` is the alias, so ln(1) matches.
        ("log(1)", None, "0 (exact)"),
        ("ln(1)", None, "0 (exact)"),
        ("log(1)", "rational", "0 (exact)"),
        ("log(1)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.log (natural) and is unconditionally inexact.
        ("log(2)", "floating-point", "0.6931471805599453 (inexact)"),
        ("log(10)", "floating-point", "2.302585092994046 (inexact)"),
        ("log(0.5)", "floating-point", "-0.6931471805599453 (inexact)"),
        # Fixed-point reduces in base 10 and sums the atanh series at the operand's
        # scale, inexact; log(2) ~ 0.6931 rounds to 1 at scale 0 (nearest whole number).
        (
            "log(2)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.6931)",
        ),
        (
            "log(100)",  # ln(100) ~ 4.6052 rounds to 5 at scale 0
            None,
            "5 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 4.6052)",
        ),
        (
            "log(2.000000)",  # math.log(2) at scale 6
            None,
            "0.693147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6931471806)",
        ),
        # ln is the same method under its alias.
        (
            "ln(2.000000)",
            None,
            "0.693147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6931471806)",
        ),
        (
            "log(10.000000)",  # math.log(10) at scale 6
            None,
            "2.302585 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 2.3025850930)",
        ),
        # An argument below 1 reduces to a negative power of ten -> a negative log.
        (
            "log(0.500000)",
            None,
            "-0.693147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.6931471806)",
        ),
        (
            "log(0.001000)",  # ln(0.001) ~ -6.9078, heavily reduced (n = -3)
            None,
            "-6.907755 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -6.9077552790)",
        ),
    ],
)
def test_log(expression, mode, value):
    assert _value(expression, mode) == value


_RATIONAL_LOG_REFUSAL = "logarithm of a non-unit rational is transcendental"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # No real log for a non-positive operand in any mode (x = 0 is -inf)...
        ("log(0)", None, "logarithm of a non-positive value"),
        ("log(-1)", None, "logarithm of a non-positive value"),
        ("log(0)", "floating-point", "logarithm of a non-positive value"),
        ("log(-2)", "floating-point", "logarithm of a non-positive value"),
        ("log(0)", "rational", "logarithm of a non-positive value"),
        # ...and rational refuses a transcendental log (no scale to round to).
        ("log(2)", "rational", _RATIONAL_LOG_REFUSAL),
        ("ln(2)", "rational", _RATIONAL_LOG_REFUSAL),
    ],
)
def test_log_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # A power of ten logs EXACTLY in every exact mode — the base-10 reduction
        # extracts the whole exponent with no rounding (log10's richer landmark).
        ("log10(1)", None, "0 (exact)"),
        ("log10(100)", None, "2 (exact)"),
        ("log10(1000)", None, "3 (exact)"),
        ("log10(0.010)", None, "-2.000 (exact)"),  # negative exponent, scale preserved
        ("log10(0.001000)", None, "-3.000000 (exact)"),
        ("log10(100)", "rational", "2 (exact)"),
        ("log10(1)", "rational", "0 (exact)"),
        ("log10(1/10)", "rational", "-1 (exact)"),  # 10**-1 as 1/denominator
        ("log10(1/100)", "rational", "-2 (exact)"),
        # binary64 uses math.log10 and is unconditionally inexact, even on a power of ten.
        ("log10(100)", "floating-point", "2.0 (inexact)"),
        ("log10(2)", "floating-point", "0.3010299956639812 (inexact)"),
        ("log10(0.001)", "floating-point", "-3.0 (inexact)"),
        # A non-power-of-ten fixed-point argument divides ln(x)/ln(10), inexact;
        # log10(2) ~ 0.3010 rounds to 0 at scale 0 (nearest whole number).
        (
            "log10(2)",
            None,
            "0 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.3010)",
        ),
        (
            "log10(2.000000)",  # math.log10(2) at scale 6
            None,
            "0.301030 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.3010299957)",
        ),
        (
            "log10(50.000000)",  # ~1.6990, between two powers of ten
            None,
            "1.698970 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.6989700043)",
        ),
    ],
)
def test_log10(expression, mode, value):
    assert _value(expression, mode) == value


_RATIONAL_LOG10_REFUSAL = "base-10 logarithm of a non-power-of-ten rational is irrational"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # Same non-positive refusal as log, in every mode...
        ("log10(0)", None, "logarithm of a non-positive value"),
        ("log10(-5)", None, "logarithm of a non-positive value"),
        ("log10(0)", "floating-point", "logarithm of a non-positive value"),
        ("log10(0)", "rational", "logarithm of a non-positive value"),
        # ...and rational refuses anything that is not an integer power of ten.
        ("log10(2)", "rational", _RATIONAL_LOG10_REFUSAL),
        ("log10(1/3)", "rational", _RATIONAL_LOG10_REFUSAL),
    ],
)
def test_log10_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # exp(0) = 1 is the one exact case in every mode (transcendental otherwise),
        # the inverse landmark of log(1) = 0.
        ("exp(0)", None, "1 (exact)"),
        ("exp(0)", "rational", "1 (exact)"),
        ("exp(0.000000)", None, "1.000000 (exact)"),  # scale preserved on the exact case
        ("exp(0)", "floating-point", "1.0 (inexact)"),
        # binary64 uses math.exp and is unconditionally inexact; exp(1) is e.
        ("exp(1)", "floating-point", "2.718281828459045 (inexact)"),
        ("exp(2)", "floating-point", "7.38905609893065 (inexact)"),
        ("exp(-1)", "floating-point", "0.36787944117144233 (inexact)"),
        # Fixed-point range-reduces by ln(2) and sums the all-plus series at the
        # operand's scale, inexact; exp(5) ~ 148.41 rounds to 148 at scale 0.
        (
            "exp(5)",
            None,
            "148 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 148.4132)",
        ),
        (
            "exp(1.000000)",  # e at scale 6
            None,
            "2.718282 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 2.7182818285)",
        ),
        # A negative argument is fine (k < 0): exp(-1) = 1/e, below 1.
        (
            "exp(-1.000000)",
            None,
            "0.367879 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.3678794412)",
        ),
        # A large argument forces many 2**k restorations; the working scale carries
        # the result's integer digits so the last place is still right.
        (
            "exp(20.000000)",
            None,
            "485165195.409790 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 485165195.4097902780)",
        ),
    ],
)
def test_exp(expression, mode, value):
    assert _value(expression, mode) == value


def test_exp_fixed_point_honours_a_raised_precision_floor():
    # min_fixed_point_precision lifts the scale: exp(1) to 10 places is e exactly
    # rounded, the value the scale-6 default advertises in its hint.
    assert _value("exp(1.0000000000)", None, 10) == "2.7182818285 (inexact, rounded to 10 decimals)"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # rational refuses a transcendental exp (no scale to round to) — exp of any
        # non-zero rational is irrational (Lindemann-Weierstrass), the exact-or-refuse
        # mirror of log's non-unit refusal.
        ("exp(1)", "rational", "exp of a non-zero rational is transcendental"),
        ("exp(-2)", "rational", "exp of a non-zero rational is transcendental"),
        ("exp(1/2)", "rational", "exp of a non-zero rational is transcendental"),
    ],
)
def test_exp_rational_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


# --- nullary constants pi() and e() (29.2 / 29.3 / 29.4) --------------------
# Zero-argument calls, the one function shape that takes the run context instead
# of operands (29.2). Both are irrational, so the per-mode story is the same as
# sqrt's: float rounds its native double (inexact), fixed-point truncates to the
# run's DERIVED scale (29.3), and rational refuses outright.


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # binary64: the nearest double, carrying the inexact flag (math.pi / math.e).
        ("pi()", "floating-point", "3.141592653589793 (inexact)"),
        ("e()", "floating-point", "2.718281828459045 (inexact)"),
        # A nullary is an ordinary atom, so it threads through surrounding arithmetic.
        ("2 * pi()", "floating-point", "6.283185307179586 (inexact)"),
    ],
)
def test_nullary_floating_point(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # rational refuses: an irrational constant has no finite fraction (like sqrt(2)).
        ("pi()", "rational", "pi is irrational; no rational value"),
        ("e()", "rational", "e is irrational; no rational value"),
        ("2 * pi()", "rational", "pi is irrational; no rational value"),
    ],
)
def test_nullary_rational_refuses(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("bare", "called"),
    [("pi", "pi()"), ("e", "e()"), ("2 * pi", "2 * pi()"), ("e ** 2", "e() ** 2")],
)
@pytest.mark.parametrize("mode", ["floating-point", "fixed-point"])
def test_bare_constant_matches_the_call_form(bare, called, mode):
    # The bare constant (29.6) is sugar for the nullary call, so it evaluates byte-for-byte
    # the same in every mode — derived fixed-point scale included (the floor of 0 here).
    assert _calc(bare, mode) == _calc(called, mode)


def test_bare_constant_rational_refuses_like_the_call():
    # Same irrational refusal as pi()/e() — the sugar changes only the syntax (29.6).
    assert _calc("pi", "rational")["error"] == "pi is irrational; no rational value"
    assert _calc("e", "rational")["error"] == "e is irrational; no rational value"


@pytest.mark.parametrize(
    ("expression", "floor", "value"),
    [
        # No literal to derive a scale from: the nullary sits at the default floor of
        # 0 decimals, the hint nudging toward min_fixed_point_precision for more (29.3).
        (
            "pi()",
            None,
            "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 3.1415)",
        ),
        (
            "e()",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.7182)",
        ),
        # A 0xffff@8 literal pushes the derived scale to 8; the nullary matches it (29.3).
        (
            "0xffff@8 + pi()",
            None,
            "3.14224800 (inexact, rounded to 8 decimals — pass min_fixed_point_precision "
            "for more; e.g. =12 → 3.142248003589)",
        ),
        # A 2.000000000 literal (9 written decimals) pushes the scale to 9.
        (
            "2.000000000 + pi()",
            None,
            "5.141592653 (inexact, rounded to 9 decimals — pass min_fixed_point_precision "
            "for more; e.g. =13 → 5.1415926535897)",
        ),
        # With no literal, the floor IS the derived scale: min_fixed_point_precision=18
        # gives the bare nullary 18 decimals — the ERC-20 idiom — the floor winning
        # where it exceeds the default 0.
        ("pi()", 18, "3.141592653589793238 (inexact, rounded to 18 decimals)"),
        ("e()", 18, "2.718281828459045235 (inexact, rounded to 18 decimals)"),
    ],
)
def test_nullary_fixed_point_precision_derivation(expression, floor, value):
    # fixed-point is the default mode (mode omitted), so this also exercises the
    # mode-without-operand dispatch (29.2): the run mode reaches the nullary.
    assert _value(expression, floor=floor) == value


# --- nullary clock reading time() (28.1) ------------------------------------
# The first clock-DEPENDENT nullary. UNLIKE the irrational constants pi()/e() a
# tick is exactly rational, so it refuses in NO mode and is inexact only in float.
# The tool path samples the LIVE realtime clock (asserted only for type/range);
# the exact per-mode and per-scale renders are pinned by injecting a fixed epoch
# through evaluate()'s now_ns hook (28.1.2).

_EPOCH_NS = 1_700_000_000_123_456_789  # a fixed instant, 1700000000.123456789 s


def _time_value(expression, mode, floor=0):
    """Evaluate `expression` at the fixed test epoch, returning the Value (28.1.2)."""
    return parse(expression).evaluate(mode, min_fixed_point_precision=floor, now_ns=_EPOCH_NS)


@pytest.mark.parametrize(
    ("floor", "mantissa", "decimals"),
    [
        # Derived scale s == the floor here (no literal). The ns reading is TRUNCATED
        # to s decimals — a resolution choice, so the render is EXACT (28.1.1).
        (0, 1_700_000_000, 0),  # whole seconds, the C library time()
        (3, 1_700_000_000_123, 3),  # tv_nsec truncated to milliseconds
        (9, 1_700_000_000_123_456_789, 9),  # the full ns reading, no truncation
        (12, 1_700_000_000_123_456_789_000, 12),  # past 9: true zeros, the clock has no finer grain
    ],
)
def test_time_fixed_point_truncates_to_the_derived_scale(floor, mantissa, decimals):
    value = _time_value("time()", Mode.FIXED_POINT, floor=floor)
    assert value.mode is Mode.FIXED_POINT
    assert value.payload == FixedPoint(mantissa, decimals)
    assert value.exact is True  # the rendered decimal IS the sampled value


@pytest.mark.parametrize(
    ("floor", "rendered"),
    [
        # The headline behaviour: the precision floor IS the clock resolution, and
        # the rendered fixed-point decimal carries it verbatim (28.1.1). _EPOCH_NS is
        # 1700000000.123456789 s, so each scale is that epoch cut at that resolution.
        (0, "1700000000"),  # whole seconds
        (3, "1700000000.123"),  # milliseconds
        (6, "1700000000.123456"),  # microseconds — TRUNCATED (rounding would give ...457)
        (9, "1700000000.123456789"),  # nanoseconds — the clock's full grain
        (12, "1700000000.123456789000"),  # beyond ns: true trailing zeros, no finer grain
    ],
)
def test_time_fixed_point_renders_epoch_at_the_requested_resolution(floor, rendered):
    # The fixed-point string a caller actually sees: epoch.<fraction> at the scale
    # set by min_fixed_point_precision (3 -> ms, 6 -> us, 9 -> ns).
    assert _time_value("time()", Mode.FIXED_POINT, floor=floor).to_string() == rendered


def test_time_floating_point_is_the_full_ns_as_an_inexact_double():
    value = _time_value("time()", Mode.FLOATING_POINT)
    assert value.mode is Mode.FLOATING_POINT
    assert value.payload == float(Fraction(_EPOCH_NS, 10**9))
    assert value.exact is False  # the only mode that rounds


def test_time_rational_is_the_full_ns_exact_and_does_not_refuse():
    # Where pi()/e() raise in rational, time() returns the exact fraction (28.1).
    value = _time_value("time()", Mode.RATIONAL)
    assert value.mode is Mode.RATIONAL
    assert value.payload == Fraction(_EPOCH_NS, 10**9)
    assert value.exact is True


def test_time_is_one_instant_per_run():
    # Every time() in an expression shares the single sampled reading (28.1.2),
    # so a self-difference is exactly zero — even off the LIVE clock (now_ns
    # defaulted), since the whole run samples once.
    value = parse("time() - time()").evaluate(Mode.RATIONAL)
    assert value.payload == Fraction(0)


def test_time_live_clock_through_the_tool_is_a_recent_epoch():
    # The functional path samples the real clock; fixed-point default scale 0 gives
    # whole seconds, exact. Assert only a sane range (this test is time-dependent).
    rendered = _value("time()")  # e.g. "1765432100 (exact)"
    seconds, annotation = rendered.split(" ", 1)
    assert annotation == "(exact)"
    assert int(seconds) > 1_700_000_000  # after 2023-11-14


def test_time_rational_through_the_tool_does_not_refuse():
    # Contrast with test_nullary_rational_refuses: a clock tick has a rational value.
    payload = _calc("time()", "rational")
    assert payload["error"] is None
    assert payload["value"] is not None
