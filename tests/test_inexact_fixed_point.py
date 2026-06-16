# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for inexact-handling on fixed-point calculations (TODO 35.2).

The companion to test_rounding_fixed_point.py: that file pins WHICH fixed-point
operations come out inexact; this one pins what the caller-supplied
``inexact_handling`` policy DOES about it. The default CONTINUE_AND_REPORT
computes and lets the verdict surface (35.2.1); ABORT_ON_INEXACT unwinds the
calculation the moment a value is inexact, with a diagnostic that names WHERE the
inexactness entered, WHAT KIND it is, and HOW MUCH was lost (35.2.2).

Scope: FIXED_POINT — the mode where inexactness is a real, decidable property of
the math (float is conservatively always-inexact, rational refuses irrationals).
Those cross-mode contrasts live in test_evaluate.py; the tool/e2e layers in
test_server.py and test_e2e.py.
"""

import pytest

from mcp_abacus.expr.nodes import EvalError
from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import InexactHandling, Mode

ABORT = InexactHandling.ABORT_ON_INEXACT
CONTINUE = InexactHandling.CONTINUE_AND_REPORT


def _eval_fp(expression: str, handling: InexactHandling = CONTINUE, floor: int = 0):
    """Parse and evaluate a whole expression in fixed-point — the calculate path."""
    return parse(expression).evaluate(Mode.FIXED_POINT, floor, inexact_handling=handling)


def _abort_fp(expression: str, floor: int = 0) -> EvalError:
    """Evaluate under ABORT_ON_INEXACT, asserting it raises; return the EvalError."""
    with pytest.raises(EvalError) as excinfo:
        _eval_fp(expression, ABORT, floor)
    return excinfo.value


# --- the default policy: continue and report (35.2.1) ----------------------
#
# CONTINUE_AND_REPORT is the historical behaviour — an inexact fixed-point result
# still computes and returns, carrying its inexact verdict, never raising.


@pytest.mark.parametrize(
    "expression, text",
    [
        ("10 / 3", "3"),  # non-terminating division, rounded at scale 0
        ("100.00 / 3", "33.33"),  # non-terminating, rounded at scale 2
        ("1.5 * 1.5", "2.2"),  # product overflows the covering scale
        ("2 ** 0.5", "1.4"),  # irrational root, covering scale 1 (from the 0.5 exponent)
    ],
)
def test_continue_and_report_returns_inexact_results(expression, text):
    result = _eval_fp(expression)  # default handling
    assert result.to_string() == text
    assert result.exact is False


def test_continue_is_the_default_when_no_policy_is_passed():
    # Omitting inexact_handling reproduces the pre-35.2 behaviour exactly.
    assert _eval_fp("10 / 3").to_string() == "3"
    assert _eval_fp("10 / 3", CONTINUE).to_string() == "3"


# --- abort on inexact: an inexact result is rejected (35.2.2) --------------
#
# ABORT_ON_INEXACT raises an EvalError the moment a fixed-point value rounds,
# tagged with the source line (here every node is line 1).


@pytest.mark.parametrize(
    "expression",
    [
        "10 / 3",  # division rounds
        "100.00 / 3",  # division rounds at scale 2
        "1.5 * 1.5",  # multiplication overflows the covering scale
        "1.5 ** 2",  # integer power overflows like the mul
        "2 ** 0.5",  # irrational root
        "sqrt(2)",  # irrational function
    ],
)
def test_abort_on_inexact_rejects_a_rounded_result(expression):
    error = _abort_fp(expression)
    assert error.line == 1
    assert "is not exact" in error.message


# --- abort leaves an exact result untouched --------------------------------
#
# The policy only fires on inexactness — an exact fixed-point calculation returns
# normally, flagged exact, exactly as under the default.


@pytest.mark.parametrize(
    "expression, text",
    [
        ("1 + 2", "3"),  # integer arithmetic, exact
        ("100.00 / 4", "25.00"),  # exact division, non-integer scale
        ("100.00 / 8", "12.50"),  # exact non-integer quotient
        ("1.50 * 1.50", "2.25"),  # product fits the covering scale
        ("0.25 ** 0.5", "0.50"),  # perfect root lands on the grid
    ],
)
def test_abort_passes_an_exact_result_through(expression, text):
    result = _eval_fp(expression, ABORT)
    assert result.to_string() == text
    assert result.exact is True


# --- WHERE: the diagnostic names the offending sub-expression (35.1.3) ------
#
# The message unparses the exact node that first went inexact — and because the
# walk is depth-first, that is the introduction SITE, not the whole expression.


def test_abort_names_the_offending_sub_expression():
    error = _abort_fp("100.00 / 3")
    assert "100.00 / 3" in error.message


def test_abort_fires_at_the_introduction_site_not_the_outer_expression():
    # (1.00 / 3.00) + 2.00 — the inner division is where inexactness ENTERS; the
    # addition above it merely inherits it, so the message points at the division.
    error = _abort_fp("(1.00 / 3.00) + 2.00")
    assert "1.00 / 3.00" in error.message
    assert "+ 2.00" not in error.message


def test_abort_carries_the_failing_line_in_a_multi_line_program():
    # The rounding happens on the second statement; the EvalError line is that line.
    # The headline shows the operands as their VALUES (35.3.1), so the variable x
    # reads as the 10 it holds, not the lexeme "x".
    error = _abort_fp("x = 10\ny = x / 3\ny")
    assert error.line == 2
    assert "10 / 3" in error.message


# --- 35.3.1: the headline always reads "<operands> = <result> is not exact" ----
#
# The first part is invariant and shows the computed NUMBERS at the active
# precision, never the source lexemes: a caller's `1 / 3` reads as `1.00 / 3.00 =
# 0.33` once the active precision is two decimals.


def test_abort_headline_renders_operands_as_values_at_active_precision():
    # min_fixed_point_precision=2 makes the active precision two decimals, so the
    # bare literals 1 and 3 show padded — exactly the user-facing example.
    error = _abort_fp("1 / 3", floor=2)
    assert error.message.startswith(
        "Inexact calculation in line 1: 1.00 / 3.00 = 0.33 is not exact."
    )


def test_abort_headline_pads_an_operand_to_the_result_precision():
    # 100.00 / 3 — the literal 3 has its own scale 0, but the quotient is carried at
    # scale 2, so the headline shows 3.00 (the operand at the ACTIVE precision), not
    # the lexeme 3.
    error = _abort_fp("100.00 / 3")
    assert error.message.startswith(
        "Inexact calculation in line 1: 100.00 / 3.00 = 33.33 is not exact."
    )


def test_abort_headline_renders_a_function_call_over_its_argument_value():
    # A function abort lays the call out over its argument's value, not a binary form.
    error = _abort_fp("sqrt(2)")
    assert error.message.startswith(
        "Inexact calculation in line 1: sqrt(2) = 1 is not exact."
    )


# --- 35.3.2: the first hint line says how to LIFT the abort -----------------
#
# After the headline, an always-present hint names the policy that computes the
# inexact result instead of rejecting it — continue-and-report.


def test_abort_message_hints_how_to_enable_inexact_calculations():
    message = _abort_fp("1.00 / 3.00").message
    assert (
        "\n - Pass inexact_handling='continue-and-report' to enable inexact calculations."
        in message
    )


# --- the conditional hint lines are deferred (35.3.3-35.3.5) ----------------
#
# The remaining hints — the residual / raise-precision steer (35.3.3), the "this
# function can't be exact" note (35.3.4), and the rational-mode-is-exact offer
# (35.3.5) — are not implemented yet, so the residual and the irrational/rational
# verdict do NOT appear in the message. Value.explain_inexact still carries that
# information (covered in test_value.py) for those points to format.


# --- min_fixed_point_precision interaction ---------------------------------
#
# The floor decides whether a division is exact, so it decides whether ABORT
# fires: a terminating quotient becomes exact with enough decimals and stops
# aborting, while a non-terminating one rounds at ANY finite scale and keeps
# aborting — the policy tracks the real exactness, not a fixed scale.


def test_a_terminating_division_stops_aborting_once_the_floor_makes_it_exact():
    # 10 / 4 == 2.5: inexact (rounds to 2) at scale 0, exact at scale 2.
    assert _abort_fp("10 / 4").line == 1  # scale 0 -> rounds -> aborts
    result = _eval_fp("10 / 4", ABORT, floor=2)  # scale 2 -> 2.50 exact -> no abort
    assert result.to_string() == "2.50" and result.exact is True


def test_a_non_terminating_division_keeps_aborting_at_any_floor():
    # 10 / 3 never terminates, so it rounds — and aborts — at every finite scale.
    assert _abort_fp("10 / 3").line == 1
    assert _abort_fp("10 / 3", floor=10).line == 1
