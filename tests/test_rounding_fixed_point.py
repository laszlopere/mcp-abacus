# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for rounding modes and round-only-at-display (TODO 34).

This is the home for everything TODO 34 adds: the exactness contract of the
`/` operator, the round-only-at-display behaviour, the selectable rounding
modes, and whichever inexact-handling policy 34.5 settles on.

Scope for now: FIXED_POINT only — the mode the round-only-at-display decision
is about. Floating-point and rational cases join here as that work lands.
"""

from fractions import Fraction

import pytest

from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import Mode


def _eval_fp(expression: str):
    """Parse and evaluate a whole expression in fixed-point — the calculate path."""
    return parse(expression).evaluate(Mode.FIXED_POINT)


# --- '/' operator exactness ------------------------------------------------
#
# Division need not be whole to be exact: `a / b` at scale s is exact iff b
# divides a·10^s. The starter set proves both an integer and a non-integer
# quotient stay exact when the division discards nothing.


@pytest.mark.parametrize(
    "expression, text, exact",
    [
        ("100.00 / 4", "25.00", True),  # integer quotient, exact
        ("100.00 / 8", "12.50", True),  # non-integer quotient, still exact
        ("100.00 / 3", "33.33", False),  # 33.333... doesn't fit scale 2 -> inexact
    ],
)
def test_fixed_point_division_exactness(expression, text, exact):
    result = _eval_fp(expression)
    assert result.to_string() == text
    assert result.exact is exact


# --- '*' operator exactness ------------------------------------------------
#
# Multiplication is exact iff the product fits the covering scale max(d1, d2).
# The same digits at a wider written scale stay exact; at a narrower one they
# round — so the WRITTEN scale, not the value, decides exactness.


@pytest.mark.parametrize(
    "expression, text, exact",
    [
        ("1.50 * 1.50", "2.25", True),  # 2.25 fits scale 2 -> exact, non-integer
        ("1.5 * 1.5", "2.2", False),  # same value, scale 1 can't hold 2.25 -> inexact
    ],
)
def test_fixed_point_multiplication_exactness(expression, text, exact):
    result = _eval_fp(expression)
    assert result.to_string() == text
    assert result.exact is exact


# --- '**' operator exactness -----------------------------------------------
#
# Power has TWO inexact paths: an integer exponent is repeated '*', so it
# overflows the covering scale just like '*'; a fractional exponent is a root,
# exact only when it lands on the grid (a perfect root), inexact when irrational.


@pytest.mark.parametrize(
    "expression, text, exact",
    [
        ("1.50 ** 2", "2.25", True),  # 2.25 fits scale 2 -> exact, non-integer
        ("0.25 ** 0.5", "0.50", True),  # sqrt(1/4) lands on the grid -> exact, non-integer
        ("1.5 ** 2", "2.2", False),  # same value as 1.50**2, scale 1 can't hold it
        ("2 ** 0.5", "1.4", False),  # sqrt(2) is irrational -> inexact
    ],
)
def test_fixed_point_power_exactness(expression, text, exact):
    result = _eval_fp(expression)
    assert result.to_string() == text
    assert result.exact is exact


# --- "how inexact": the stored quantization error (34.5.2) ------------------
#
# A rounded algebraic op carries the EXACT signed residual stored - true on
# Value.error, bounded by 1/2 ULP. Exact values carry None; so does an irrational
# root, whose true value is not a clean rational to subtract.


@pytest.mark.parametrize(
    "expression, error",
    [
        ("100.00 / 8", None),  # exact -> nothing rounded, no residual
        ("100.00 / 3", Fraction(-1, 300)),  # 33.33 - 100/3, rounds DOWN -> negative
        ("1.5 * 1.5", Fraction(-1, 20)),  # 2.2 - 2.25, exactly -1/2 ULP at scale 1
        ("1.5 ** 2", Fraction(-1, 20)),  # integer power overflows like the mul
        ("2 ** 0.5", None),  # irrational root -> no exact-rational residual
        # The POSITIVE branch: a value that rounds UP carries a positive residual. The
        # mirror of the cases above, which all round down — without these the sign of
        # `error` is only ever exercised negative.
        ("3.0 / 4.0", Fraction(1, 20)),  # 0.8 - 0.75, +1/2 ULP (tie rounds up to even 8)
        ("1.5 ** 3", Fraction(1, 40)),  # 3.4 - 3.375, integer power rounding up
        # A STRICTLY-INTERIOR residual (|error| < 1/2 ULP), not a tie: 0.2 - 1.1/7 = 3/70
        # ~= 0.0429 < 0.05. Catches an off-by-one in the half-ULP bound the tie cases miss.
        ("1.1 / 7", Fraction(3, 70)),
        # Covering-scale rounding between DIFFERENT operand scales (2 and 1 -> scale 2).
        ("1.25 * 1.5", Fraction(1, 200)),  # 1.88 - 1.875, rounds up at the covering scale
        # A negative divisor that rounds: the den<0 sign-normalization signs both the
        # quotient and its residual. -0.33 - (-1/3) = +1/300.
        ("1.00 / -3.00", Fraction(1, 300)),
    ],
)
def test_fixed_point_stores_quantization_error(expression, error):
    result = _eval_fp(expression)
    assert result.error == error
    # An exact value never carries an error; an inexact algebraic op always does.
    if error is not None:
        assert abs(error) <= Fraction(1, 2) * Fraction(1, 10**result.payload.decimals)
