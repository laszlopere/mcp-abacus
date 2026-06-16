# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for rounding modes and round-only-at-display (TODO 34).

This is the home for everything TODO 34 adds: the exactness contract of the
`/` operator, the round-only-at-display behaviour, the selectable rounding
modes, and whichever inexact-handling policy 34.5 settles on.

Scope for now: FIXED_POINT only — the mode the round-only-at-display decision
is about. Floating-point and rational cases join here as that work lands.
"""

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
