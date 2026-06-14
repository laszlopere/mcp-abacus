# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver engine, built up item by item (TODO 31)."""

import math

import pytest

from mcp_abacus.expr.nodes import EvalError
from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import Mode, Value
from mcp_abacus.solver import (
    Goal,
    SolverError,
    SolverType,
    objective,
    resolve_goal,
    resolve_type,
    search,
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


# --- the search engine (31.7) -------------------------------------------------


def test_solve_finds_a_root_in_floating_point():
    # x**2 - 2 == 0 over [0, 2] -> sqrt(2); the expression is ~0 at the solution.
    result = search(
        parse("x**2 - 2"), "x", 0.0, 2.0, Mode.FLOATING_POINT, 0, SolverType.SOLVE, None
    )
    assert result.solution.to_float() == pytest.approx(2**0.5, abs=1e-9)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-6)
    assert result.type is SolverType.SOLVE and result.goal is None
    assert result.iterations > 0


def test_solve_finds_an_exact_root_on_the_fixed_point_grid():
    # 2*x - 3 == 0 over [0, 3] -> 1.5, exactly representable at scale 1, so the
    # expression evaluates to an EXACT zero there (grid polish lands on it, 31.7).
    result = search(parse("2*x - 3"), "x", 0.0, 3.0, Mode.FIXED_POINT, 1, SolverType.SOLVE, None)
    assert result.solution.to_string() == "1.5"
    assert result.value.to_string() == "0.0"
    assert result.value.exact is True


def test_optimise_minimise_finds_the_low_point():
    # (x - 3)**2 is unimodal with its minimum at x == 3, value 0.
    result = search(
        parse("(x - 3)**2"), "x", 0.0, 5.0, Mode.FLOATING_POINT, 0,
        SolverType.OPTIMISE, Goal.MINIMISE,
    )
    assert result.solution.to_float() == pytest.approx(3.0, abs=1e-6)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-9)
    assert result.goal is Goal.MINIMISE


def test_optimise_maximise_finds_the_high_point():
    # 5 - (x - 1)**2 peaks at x == 1 with value 5; maximise drives there.
    result = search(
        parse("5 - (x - 1)**2"), "x", -2.0, 4.0, Mode.FLOATING_POINT, 0,
        SolverType.OPTIMISE, Goal.MAXIMISE,
    )
    assert result.solution.to_float() == pytest.approx(1.0, abs=1e-5)
    assert result.value.to_float() == pytest.approx(5.0, abs=1e-9)
    assert result.goal is Goal.MAXIMISE


def test_solve_with_constants_set_by_assignments():
    # The program sets r, p by assignment; n is the free unknown. Solve the
    # compound-interest break-even p*(1+r)**n == 2000 for n over [0, 100].
    program = "r = 0.05\np = 1000\np * (1 + r)**n - 2000"
    result = search(parse(program), "n", 0.0, 100.0, Mode.FLOATING_POINT, 0, SolverType.SOLVE, None)
    # 1000 * 1.05**n == 2000 -> n == ln 2 / ln 1.05 ~ 14.2067.
    assert result.solution.to_float() == pytest.approx(math.log(2) / math.log(1.05), abs=1e-6)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-6)


def test_solve_reports_no_solution_when_zero_is_unreachable():
    # x**2 + 1 is never zero on [0, 2]; its closest |expr| is 1 at x == 0.
    with pytest.raises(SolverError) as excinfo:
        search(parse("x**2 + 1"), "x", 0.0, 2.0, Mode.FLOATING_POINT, 0, SolverType.SOLVE, None)
    message = excinfo.value.message
    assert "No solution" in message and "closest" in message


def test_domain_failures_are_penalised_not_fatal():
    # sqrt(x) - 1 == 0 over a bracket that dips below 0: the negative side raises a
    # domain error per candidate (penalised +inf), yet the root at x == 1 is found.
    result = search(
        parse("sqrt(x) - 1"), "x", -1.0, 4.0, Mode.FLOATING_POINT, 0, SolverType.SOLVE, None
    )
    assert result.solution.to_float() == pytest.approx(1.0, abs=1e-6)


def test_unset_constant_surfaces_as_an_eval_error():
    # `a` is neither the unknown nor assigned: it fails at EVERY candidate, so it is
    # a structural user error (EvalError), not a region for the search to avoid.
    with pytest.raises(EvalError) as excinfo:
        search(parse("a * x - 1"), "x", 0.0, 2.0, Mode.FLOATING_POINT, 0, SolverType.SOLVE, None)
    assert "undefined variable: a" in excinfo.value.message
