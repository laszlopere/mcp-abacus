# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for variables (TODO 30): assignment, reference, and multi-statement runs.

End to end through ``parse`` then ``evaluate``: an ``x = expr`` assignment binds a
name in the run's VariableStore (30.1/30.2) and yields the bound Value; a bare
``x`` reads it back (30.4); newline-separated statements parse to a Sequence whose
result is the LAST statement's Value (30.5), the leading ones running for their
bindings. An unset reference raises EvalError tagged with the failing node's line.

The store is pure value passthrough — it keeps each Value verbatim, with no mode
or scale of its own — so a reference returns the SAME object the assignment
produced, identical across every mode.
"""

from fractions import Fraction

import pytest

from mcp_abacus.expr.nodes import Assign, EvalError, Sequence, Var
from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import (
    Mode,
    UndefinedVariableError,
    VariableStore,
)

MODES = [Mode.FLOATING_POINT, Mode.FIXED_POINT, Mode.RATIONAL]


# --- the store itself (30.1) -----------------------------------------------


def test_store_set_then_get_round_trips_the_value():
    store = VariableStore()
    value = parse("2 + 3").evaluate(Mode.RATIONAL)
    store.set("x", value)
    assert store.get("x") is value  # the same object, verbatim


def test_store_set_overwrites_a_previous_binding():
    store = VariableStore()
    store.set("x", parse("1").evaluate(Mode.RATIONAL))
    store.set("x", parse("2").evaluate(Mode.RATIONAL))
    assert store.get("x").payload == Fraction(2)


def test_store_get_on_unset_name_raises_carrying_the_name():
    with pytest.raises(UndefinedVariableError) as excinfo:
        VariableStore().get("nope")
    assert excinfo.value.name == "nope"


# --- parsing the variable forms (30.3 / 30.4 / 30.5) -----------------------


def test_assignment_parses_to_an_assign_node():
    tree = parse("x = 1 + 2")
    assert isinstance(tree, Assign)
    assert tree.name == "x"
    assert tree.expr == parse("1 + 2")


def test_bare_name_parses_to_a_var_reference():
    tree = parse("x")
    assert tree == Var("x", line=1)


def test_constant_lookalike_names_still_parse_to_a_var():
    # Only the EXACT names pi/e are reserved constants (29.6) — names that merely
    # contain them stay ordinary variables, so the sugar is not over-broad.
    assert parse("pie") == Var("pie", line=1)
    assert parse("ex") == Var("ex", line=1)
    assert parse("epsilon") == Var("epsilon", line=1)


def test_newline_separated_statements_parse_to_a_sequence():
    tree = parse("x = 1\nx + 2")
    assert isinstance(tree, Sequence)
    assert len(tree.statements) == 2
    assert isinstance(tree.statements[0], Assign)
    assert isinstance(tree.statements[1], type(parse("x + 2")))


def test_single_statement_is_not_wrapped_in_a_sequence():
    # No Sequence for a one-statement input — the bare node stands alone (30.5).
    assert not isinstance(parse("x = 1"), Sequence)


# --- evaluating assignment + reference (30.6) ------------------------------


def test_assignment_yields_its_value():
    # x = 2 + 3 evaluates the RHS, binds x, and the assignment's OWN value is 5.
    assert parse("x = 2 + 3").evaluate(Mode.RATIONAL).payload == Fraction(5)


def test_reference_reads_back_the_bound_value():
    result = parse("x = 41\nx + 1").evaluate(Mode.RATIONAL)
    assert result.payload == Fraction(42)


def test_multi_line_program_returns_the_last_statements_value():
    # Leading statements run for their bindings; the last line is the result, the
    # way a REPL echoes its final line. y depends on x set two lines earlier.
    result = parse("x = 10\ny = x * 2\ny + 1").evaluate(Mode.RATIONAL)
    assert result.payload == Fraction(21)


def test_reassignment_within_a_run_overwrites():
    result = parse("x = 1\nx = x + 5\nx").evaluate(Mode.RATIONAL)
    assert result.payload == Fraction(6)


def test_undefined_reference_raises_evalerror_tagged_with_its_line():
    # The unset name is on line 2; the EvalError carries that line, not line 1.
    with pytest.raises(EvalError) as excinfo:
        parse("x = 1\nz").evaluate(Mode.RATIONAL)
    assert excinfo.value.line == 2
    assert "undefined variable: z" in str(excinfo.value)


def test_bindings_do_not_leak_across_separate_evaluate_runs():
    # Each evaluate() builds a fresh store (30.2): x set in one run is gone in the next.
    parse("x = 5").evaluate(Mode.RATIONAL)
    with pytest.raises(EvalError):
        parse("x").evaluate(Mode.RATIONAL)


# --- per-mode value passthrough (30.7) -------------------------------------
# The store holds Values verbatim — no mode or scale of its own — so a reference
# returns exactly what the assignment computed, in whatever mode the run uses.


@pytest.mark.parametrize("mode", MODES)
def test_reference_returns_the_assignments_value_verbatim(mode):
    # x = 2 + 3 then x: the read-back equals evaluating 2 + 3 directly in that mode.
    read_back = parse("x = 2 + 3\nx").evaluate(mode)
    direct = parse("2 + 3").evaluate(mode)
    assert read_back == direct
    assert read_back.mode is mode


@pytest.mark.parametrize("mode", MODES)
def test_passthrough_keeps_the_exact_same_object(mode):
    # Identity, not just equality: the Var node hands back the very Value the Assign
    # stored, proving the store re-coerces nothing (no mode/scale of its own).
    tree = parse("x = 1 + 1\nx")
    tree.evaluate(mode)
    assert isinstance(tree, Sequence)
    assign, ref = tree.statements
    assert isinstance(assign, Assign) and isinstance(ref, Var)
    assert ref.value is assign.value


def test_fixed_point_passthrough_preserves_scale_and_inexactness():
    # A reference carries the stored Value's scale/exactness through unchanged: an
    # inexact 1/3 at the run's floor stays inexact at that scale when read back.
    read_back = parse("x = 1 / 3\nx").evaluate(Mode.FIXED_POINT, 5)
    direct = parse("1 / 3").evaluate(Mode.FIXED_POINT, 5)
    assert read_back == direct
    assert read_back.to_string() == "0.33333"
    assert not read_back.exact
    assert read_back.precision() == 5
