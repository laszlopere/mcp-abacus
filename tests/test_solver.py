# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver engine, built up item by item (TODO 31)."""

import pytest

from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import Mode, Value
from mcp_abacus.solver import (
    Goal,
    SolverError,
    SolverType,
    objective,
    resolve_goal,
    resolve_type,
    validate_bracket,
    validate_unknown,
)


def test_type_inferred_as_solve_when_no_type_and_no_goal():
    assert resolve_type(None, None) is SolverType.SOLVE


def test_type_inferred_as_optimise_when_a_goal_is_given():
    assert resolve_type(None, "minimise") is SolverType.OPTIMISE


def test_explicit_solve_without_a_goal_is_accepted():
    assert resolve_type("solve", None) is SolverType.SOLVE


def test_explicit_optimise_with_a_goal_is_accepted():
    assert resolve_type("optimise", "maximise") is SolverType.OPTIMISE


def test_unknown_type_lists_the_valid_types():
    with pytest.raises(SolverError) as excinfo:
        resolve_type("root-find", None)
    message = excinfo.value.message
    assert "Unknown solver type" in message
    assert "solve" in message and "optimise" in message


def test_optimise_requires_a_goal():
    with pytest.raises(SolverError) as excinfo:
        resolve_type("optimise", None)
    assert "requires a goal" in excinfo.value.message


def test_solve_forbids_a_goal():
    with pytest.raises(SolverError) as excinfo:
        resolve_type("solve", "minimise")
    assert "does not take a goal" in excinfo.value.message


def test_unknown_that_occurs_as_a_reference_is_accepted():
    validate_unknown(parse("x**2 - 2"), "x")  # does not raise


def test_unknown_alongside_assigned_constants_is_accepted():
    # The constants r, p are set by assignments; n is the free unknown.
    validate_unknown(parse("r = 0.05\np = 1000\np * (1 + r)**n - 2000"), "n")


def test_unknown_that_is_an_assignment_target_is_rejected():
    with pytest.raises(SolverError) as excinfo:
        validate_unknown(parse("x = 5\nx + 1"), "x")
    assert "computed constant" in excinfo.value.message


def test_unknown_that_does_not_occur_is_rejected():
    with pytest.raises(SolverError) as excinfo:
        validate_unknown(parse("y + 1"), "x")
    assert "does not occur" in excinfo.value.message


def test_no_goal_resolves_to_none_for_a_solve():
    assert resolve_goal(None) is None


def test_canonical_goals_resolve():
    assert resolve_goal("minimise") is Goal.MINIMISE
    assert resolve_goal("maximise") is Goal.MAXIMISE


def test_goal_aliases_resolve():
    assert resolve_goal("min") is Goal.MINIMISE
    assert resolve_goal("minimize") is Goal.MINIMISE
    assert resolve_goal("max") is Goal.MAXIMISE
    assert resolve_goal("maximize") is Goal.MAXIMISE


def test_unknown_goal_lists_valid_goals():
    with pytest.raises(SolverError) as excinfo:
        resolve_goal("biggest")
    message = excinfo.value.message
    assert "Unknown goal" in message
    assert "minimise" in message and "maximise" in message


def test_objective_for_solve_is_the_absolute_value():
    negative = Value.from_lexeme("3", Mode.FLOATING_POINT).neg()  # -3.0
    assert objective(negative, None) == negative.abs_()


def test_objective_for_minimise_is_the_value_itself():
    value = Value.from_lexeme("3", Mode.FLOATING_POINT)
    assert objective(value, Goal.MINIMISE) == value


def test_objective_for_maximise_is_the_negation():
    value = Value.from_lexeme("3", Mode.FLOATING_POINT)
    assert objective(value, Goal.MAXIMISE) == value.neg()


def test_proper_bracket_is_accepted():
    validate_bracket(0.0, 2.0)  # does not raise


def test_empty_bracket_is_rejected():
    with pytest.raises(SolverError) as excinfo:
        validate_bracket(2.0, 2.0)
    assert "bracket is empty" in excinfo.value.message


def test_inverted_bracket_is_rejected():
    with pytest.raises(SolverError) as excinfo:
        validate_bracket(3.0, 1.0)
    assert "must be below upper" in excinfo.value.message
