# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for in-node evaluation (TODO 18): evaluate(), EvalError, stored values."""

from fractions import Fraction

import pytest

from mcp_abacus.expr.nodes import BinOp, EvalError, FuncCall, Number, UnaryOp
from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import InexactHandling, Mode


def _example_tree(line: int = 1) -> BinOp:
    """The CORE CONCEPT example "1 + 2 * 10**3", every node at the given line."""
    return BinOp(
        "+",
        Number("1", line=line),
        BinOp(
            "*",
            Number("2", line=line),
            BinOp("**", Number("10", line=line), Number("3", line=line), line=line),
            line=line,
        ),
        line=line,
    )


def test_example_tree_rational():
    result = _example_tree().evaluate(Mode.RATIONAL)
    assert result.payload == Fraction(2001)
    assert result.exact


def test_example_tree_floating_point():
    result = _example_tree().evaluate(Mode.FLOATING_POINT)
    assert result.payload == 2001.0
    assert not result.exact


def test_double_star_is_power():
    power = BinOp("**", Number("10", line=1), Number("3", line=1), line=1)
    assert power.evaluate(Mode.RATIONAL).payload == Fraction(1000)  # 10 ** 3 == 1000


def test_unary_ops_evaluate():
    assert UnaryOp("-", Number("2", line=1), line=1).evaluate(Mode.RATIONAL).payload == Fraction(-2)
    assert UnaryOp("+", Number("2", line=1), line=1).evaluate(Mode.RATIONAL).payload == Fraction(2)


def test_value_starts_none_and_every_node_stores_its_result():
    tree = _example_tree()
    nodes = [tree, tree.left, tree.right, tree.right.left, tree.right.right]
    assert all(n.value is None for n in nodes)

    tree.evaluate(Mode.RATIONAL)
    mul = tree.right
    assert isinstance(mul, BinOp)
    power = mul.right
    assert isinstance(power, BinOp)
    assert tree.value is not None and tree.value.payload == Fraction(2001)
    assert mul.value is not None and mul.value.payload == Fraction(2000)
    assert power.value is not None and power.value.payload == Fraction(1000)
    assert tree.left.value is not None and tree.left.value.payload == Fraction(1)


def test_reevaluation_overwrites_stored_values():
    # ONE value, not per-mode (18.5): a calculation runs in one type at a time.
    tree = _example_tree()
    tree.evaluate(Mode.RATIONAL)
    assert tree.value is not None and tree.value.payload == Fraction(2001)
    tree.evaluate(Mode.FLOATING_POINT)
    assert tree.value is not None and tree.value.payload == 2001.0
    assert isinstance(tree.value.payload, float)


def test_eval_error_carries_failing_node_line():
    division = BinOp("/", Number("1", line=2), Number("0", line=3), line=5)
    with pytest.raises(EvalError) as excinfo:
        division.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 5  # the op that failed, not its operands


def test_eval_error_line_survives_outer_nodes():
    inner = BinOp("/", Number("1", line=2), Number("0", line=2), line=2)
    outer = BinOp("+", Number("1", line=7), inner, line=7)
    with pytest.raises(EvalError) as excinfo:
        outer.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 2  # innermost failing node wins (18.4)


def test_eval_error_on_mode_specific_domain_error():
    power = BinOp("**", Number("2", line=4), Number("0.5", line=4), line=4)
    with pytest.raises(EvalError) as excinfo:
        power.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 4
    assert power.evaluate(Mode.FLOATING_POINT).payload == 2**0.5  # other modes unaffected


# --- function refusals carry the right line (22.5) -------------------------
# Value.sqrt raises NotRepresentableError (an ArithmeticError), so the FuncCall
# node wraps it into EvalError carrying the call's OWN line — the same path as
# any other domain error (18.4). abs never refuses, so only sqrt is exercised.


def test_sqrt_negative_refusal_carries_the_call_line():
    # The negative operand sits on a different line than the call to prove the line
    # reported is the failing sqrt node's, not its argument's.
    call = FuncCall("sqrt", (UnaryOp("-", Number("4", line=2), line=2),), line=5)
    with pytest.raises(EvalError) as excinfo:
        call.evaluate(Mode.FIXED_POINT)
    assert excinfo.value.line == 5
    assert "negative" in str(excinfo.value)


def test_sqrt_rational_irrational_refusal_carries_the_call_line():
    call = FuncCall("sqrt", (Number("2", line=6),), line=6)
    with pytest.raises(EvalError) as excinfo:
        call.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 6
    assert "irrational" in str(excinfo.value)


def test_sqrt_refusal_line_survives_parse_and_outer_nodes():
    # End to end: the sqrt on the second source line refuses, and the EvalError
    # carries line 2 even though the addition wrapping it begins on line 1. The
    # expression spans lines inside parens (the only way now, 30.5).
    with pytest.raises(EvalError) as excinfo:
        parse("(1 +\nsqrt(-4))").evaluate(Mode.FIXED_POINT)
    assert excinfo.value.line == 2
    with pytest.raises(EvalError) as excinfo:
        parse("(1 +\nsqrt(2))").evaluate(Mode.RATIONAL)  # irrational rational root
    assert excinfo.value.line == 2


def test_sin_rational_refusal_carries_the_call_line():
    # sin of a non-zero rational is irrational and refuses; the EvalError carries
    # the call node's own line, the same path as sqrt's domain refusals.
    call = FuncCall("sin", (Number("1", line=4),), line=4)
    with pytest.raises(EvalError) as excinfo:
        call.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 4
    assert "irrational" in str(excinfo.value)


def test_cos_rational_refusal_carries_the_call_line():
    # cos mirrors sin: a non-zero rational is irrational and refuses on its own line.
    call = FuncCall("cos", (Number("1", line=4),), line=4)
    with pytest.raises(EvalError) as excinfo:
        call.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 4
    assert "irrational" in str(excinfo.value)


def test_tan_rational_refusal_carries_the_call_line():
    # tan (== sin/cos) mirrors sin/cos: a non-zero rational is irrational and
    # refuses on the call node's own line.
    call = FuncCall("tan", (Number("1", line=4),), line=4)
    with pytest.raises(EvalError) as excinfo:
        call.evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 4
    assert "irrational" in str(excinfo.value)


def test_stored_value_does_not_affect_equality_or_hash():
    evaluated = _example_tree()
    evaluated.evaluate(Mode.RATIONAL)
    fresh = _example_tree()
    assert evaluated == fresh
    assert hash(evaluated) == hash(fresh)


def test_pretty_appends_values_after_evaluation():
    tree = _example_tree()
    tree.evaluate(Mode.RATIONAL)
    expected = (
        "BINARY_ADD Value = 2001 (rational, exact)\n"
        '  LITERAL "1" Value = 1 (rational, exact)\n'
        "  BINARY_MUL Value = 2000 (rational, exact)\n"
        '    LITERAL "2" Value = 2 (rational, exact)\n'
        "    BINARY_POW Value = 1000 (rational, exact)\n"
        '      LITERAL "10" Value = 10 (rational, exact)\n'
        '      LITERAL "3" Value = 3 (rational, exact)'
    )
    assert tree.pretty() == expected


def test_pretty_unevaluated_unchanged():
    assert " = " not in _example_tree().pretty()


# --- min_fixed_point_precision floor threading (25.2.1) --------------------


def _division(dividend: str, divisor: str) -> BinOp:
    return BinOp("/", Number(dividend, line=1), Number(divisor, line=1), line=1)


def test_floor_propagates_through_division_keeping_more_decimals():
    # The TODO 25.2 example: 928347569 / 2345 rounds to a whole number at the
    # default scale, but the floor carries 4 decimals through to the result.
    bare = _division("928347569", "2345").evaluate(Mode.FIXED_POINT)
    assert bare.to_string() == "395884" and not bare.exact

    floored = _division("928347569", "2345").evaluate(Mode.FIXED_POINT, 4)
    assert floored.to_string() == "395883.8247"
    assert not floored.exact
    assert floored.precision() == 4


def test_floor_threads_to_every_literal_in_the_subtree():
    # 1 / 3 + 1 / 3 — the floor must reach BOTH divisions' operands, not just one,
    # so each rounds at scale 5 and their sum is reported at scale 5.
    tree = BinOp("+", _division("1", "3"), _division("1", "3"), line=1)
    result = tree.evaluate(Mode.FIXED_POINT, 5)
    assert result.to_string() == "0.66666"  # 0.33333 + 0.33333
    assert result.precision() == 5


def test_floor_keeps_an_exact_result_exact():
    # A division that comes out exactly still respects the floor's scale and stays
    # flagged exact (the padding zeros lose nothing): 10 / 4 == 2.5 -> 2.50000.
    result = _division("10", "4").evaluate(Mode.FIXED_POINT, 5)
    assert result.to_string() == "2.50000"
    assert result.exact
    assert result.precision() == 5


def test_floor_defaults_to_zero():
    # Omitting the floor reproduces the unfloored result exactly (back-compat).
    assert _division("10", "3").evaluate(Mode.FIXED_POINT).to_string() == "3"


def test_floor_does_not_disturb_modes_without_a_scale():
    # Passing a floor under floating-point / rational is inert — same value.
    assert _division("1", "2").evaluate(Mode.FLOATING_POINT, 9).payload == 0.5
    assert _division("1", "3").evaluate(Mode.RATIONAL, 9).payload == Fraction(1, 3)


# --- abort on inexact: the cross-mode mechanism (35.2.2) -------------------
# The caller-supplied InexactHandling threads down the walk in EVERY mode;
# ABORT_ON_INEXACT unwinds the moment a value is inexact. The fixed-point DEPTH
# (the diagnostic's where/kind/magnitude, the introduction site, the floor
# interaction) lives in test_inexact_fixed_point.py; these pin the bits that are
# specific to float and rational, where exactness behaves differently.

ABORT = InexactHandling.ABORT_ON_INEXACT


def test_abort_on_inexact_default_is_continue_and_report():
    # The default never rejects — an inexact fixed-point division still returns.
    result = _division("10", "3").evaluate(Mode.FIXED_POINT)
    assert result.to_string() == "3" and not result.exact


def test_abort_in_floating_point_fires_on_the_first_inexact_value():
    # Float reports every value inexact, so abort trips on the first literal. The
    # steer toward an exact type is a deferred hint (35.3.2/35.3.4); the headline
    # (35.3.1) just names the line and the value that went inexact.
    with pytest.raises(EvalError) as excinfo:
        Number("1.5", line=1).evaluate(Mode.FLOATING_POINT, inexact_handling=ABORT)
    assert excinfo.value.message.startswith(
        "Inexact calculation in line 1: 1.5 = 1.5 is not exact."
    )


def test_abort_never_trips_in_rational_mode_for_an_exact_program():
    # Rational arithmetic does not round, so 1/3 stays exact and the abort is inert.
    result = _division("1", "3").evaluate(Mode.RATIONAL, inexact_handling=ABORT)
    assert result.payload == Fraction(1, 3) and result.exact
