# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver engine, built up item by item (TODO 31); vocabulary reworked in TODO 32."""

import math

import pytest

from mcp_abacus.expr.nodes import EvalError
from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import Mode, Value
from mcp_abacus.solver import (
    Algorithm,
    Objective,
    SolverError,
    fold_objective,
    nelder_mead,
    resolve_algorithm,
    resolve_objective,
    search,
    validate_bracket,
    validate_unknown,
)


def _annotated(description: str, code: str) -> str:
    """Frame an abacus `code` snippet with a `#` comment header from `description`.

    The lexer strips the comments before evaluation, so the solver sees only the
    code; under ./run-tests.sh --human-readable the header is what makes each
    printed "Code" block state WHAT the program searches for and what it expects.
    """
    header = "\n".join(f"# {line}".rstrip() for line in description.strip("\n").splitlines())
    return f"#\n{header}\n#\n\n{code}"


def test_objective_defaults_to_find_root_when_omitted():
    assert resolve_objective(None) is Objective.FIND_ROOT


def test_canonical_objectives_resolve():
    assert resolve_objective("find-root") is Objective.FIND_ROOT
    assert resolve_objective("find-minimum") is Objective.FIND_MINIMUM
    assert resolve_objective("find-maximum") is Objective.FIND_MAXIMUM


def test_objective_aliases_resolve():
    # The pre-32 spellings still resolve to the canonical objective (never surfaced).
    assert resolve_objective("solve") is Objective.FIND_ROOT
    assert resolve_objective("minimise") is Objective.FIND_MINIMUM
    assert resolve_objective("minimize") is Objective.FIND_MINIMUM
    assert resolve_objective("min") is Objective.FIND_MINIMUM
    assert resolve_objective("maximise") is Objective.FIND_MAXIMUM
    assert resolve_objective("max") is Objective.FIND_MAXIMUM


def test_unknown_objective_lists_the_valid_objectives():
    with pytest.raises(SolverError) as excinfo:
        resolve_objective("optimise")  # bare optimise is no longer a valid objective
    message = excinfo.value.message
    assert "Unknown objective" in message
    assert "find-root" in message and "find-minimum" in message and "find-maximum" in message


def test_unknown_that_occurs_as_a_reference_is_accepted():
    validate_unknown(parse("x**2 - 2"), "x")  # does not raise


def test_unknown_alongside_assigned_constants_is_accepted():
    # The constants r, p are set by assignments; n is the free unknown. Inline '#'
    # comments document each line and are stripped by the lexer before parsing.
    validate_unknown(parse("r = 0.05  # rate\np = 1000  # principal\np * (1 + r)**n - 2000"), "n")


def test_unknown_that_is_an_assignment_target_is_rejected():
    with pytest.raises(SolverError) as excinfo:
        validate_unknown(parse("x = 5\nx + 1"), "x")
    assert "computed constant" in excinfo.value.message


def test_unknown_that_does_not_occur_is_rejected():
    with pytest.raises(SolverError) as excinfo:
        validate_unknown(parse("y + 1"), "x")
    assert "does not occur" in excinfo.value.message


def test_fold_objective_for_find_root_is_the_absolute_value():
    negative = Value.from_lexeme("3", Mode.FLOATING_POINT).neg()  # -3.0
    assert fold_objective(negative, Objective.FIND_ROOT) == negative.abs_()


def test_fold_objective_for_find_minimum_is_the_value_itself():
    value = Value.from_lexeme("3", Mode.FLOATING_POINT)
    assert fold_objective(value, Objective.FIND_MINIMUM) == value


def test_fold_objective_for_find_maximum_is_the_negation():
    value = Value.from_lexeme("3", Mode.FLOATING_POINT)
    assert fold_objective(value, Objective.FIND_MAXIMUM) == value.neg()


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


def test_find_root_in_floating_point():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2]\n"
        "expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    result = search(parse(program), "x", 0.0, 2.0, Mode.FLOATING_POINT, 0, Objective.FIND_ROOT)
    assert result.solution.to_float() == pytest.approx(2**0.5, abs=1e-9)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-6)
    assert result.objective is Objective.FIND_ROOT
    assert result.algorithm == "golden-section-search"
    assert result.iterations > 0


def test_find_root_finds_an_exact_root_on_the_fixed_point_grid():
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, so the expression\n"
        "evaluates to an EXACT zero there (grid polish, 31.7)",
        "2*x - 3",
    )
    result = search(parse(program), "x", 0.0, 3.0, Mode.FIXED_POINT, 1, Objective.FIND_ROOT)
    assert result.solution.to_string() == "1.5"
    assert result.value.to_string() == "0.0"
    assert result.value.exact is True


def test_find_minimum_finds_the_low_point():
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5]\nunimodal; expect x = 3, value 0 (the low point)",
        "(x - 3)**2",
    )
    result = search(parse(program), "x", 0.0, 5.0, Mode.FLOATING_POINT, 0, Objective.FIND_MINIMUM)
    assert result.solution.to_float() == pytest.approx(3.0, abs=1e-6)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-9)
    assert result.objective is Objective.FIND_MINIMUM


def test_find_maximum_finds_the_high_point():
    program = _annotated(
        "find-maximum of 5 - (x - 1)**2 over [-2, 4]\nexpect x = 1, value 5 (the peak)",
        "5 - (x - 1)**2",
    )
    result = search(parse(program), "x", -2.0, 4.0, Mode.FLOATING_POINT, 0, Objective.FIND_MAXIMUM)
    assert result.solution.to_float() == pytest.approx(1.0, abs=1e-5)
    assert result.value.to_float() == pytest.approx(5.0, abs=1e-9)
    assert result.objective is Objective.FIND_MAXIMUM


def test_find_root_with_constants_set_by_assignments():
    # r, p are set by assignment; n is the free unknown. The header is a full-line
    # comment block, and each assignment also carries an inline comment — all stripped
    # by the lexer, so the solver sees only the three statements.
    program = _annotated(
        "find-root for n: compound-interest break-even over [0, 100]\n"
        "1000 * 1.05**n == 2000  ->  n = ln2 / ln1.05 ~ 14.2067",
        "r = 0.05  # annual rate\np = 1000  # principal\np * (1 + r)**n - 2000  # solve for n",
    )
    result = search(parse(program), "n", 0.0, 100.0, Mode.FLOATING_POINT, 0, Objective.FIND_ROOT)
    assert result.solution.to_float() == pytest.approx(math.log(2) / math.log(1.05), abs=1e-6)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-6)


def test_find_root_reports_no_solution_when_zero_is_unreachable():
    program = _annotated(
        "find-root of x**2 + 1 over [0, 2]\n"
        "no solution: never zero; the closest |expr| is 1 at x = 0",
        "x**2 + 1",
    )
    with pytest.raises(SolverError) as excinfo:
        search(parse(program), "x", 0.0, 2.0, Mode.FLOATING_POINT, 0, Objective.FIND_ROOT)
    message = excinfo.value.message
    assert "No solution" in message and "closest" in message


def test_domain_failures_are_penalised_not_fatal():
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] (bracket dips below 0)\n"
        "the negative side raises a domain error per candidate (penalised\n"
        "+inf), yet expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    result = search(parse(program), "x", -1.0, 4.0, Mode.FLOATING_POINT, 0, Objective.FIND_ROOT)
    assert result.solution.to_float() == pytest.approx(1.0, abs=1e-6)


def test_unset_constant_surfaces_as_an_eval_error():
    program = _annotated(
        "find-root of a * x - 1 over [0, 2]\n"
        "`a` is neither the unknown nor assigned: it fails at EVERY candidate,\n"
        "so it is a structural user error (EvalError), not a region to avoid",
        "a * x - 1",
    )
    with pytest.raises(EvalError) as excinfo:
        search(parse(program), "x", 0.0, 2.0, Mode.FLOATING_POINT, 0, Objective.FIND_ROOT)
    assert "undefined variable: a" in excinfo.value.message


# --- algorithm resolution (33.14) ---------------------------------------------


def test_algorithm_defaults_to_golden_section_when_omitted():
    assert resolve_algorithm(None) is Algorithm.GOLDEN_SECTION


def test_canonical_algorithms_resolve():
    assert resolve_algorithm("golden-section-search") is Algorithm.GOLDEN_SECTION
    assert resolve_algorithm("nelder-mead") is Algorithm.NELDER_MEAD


def test_algorithm_aliases_resolve():
    # Never-surfaced spellings still resolve to the canonical engine.
    assert resolve_algorithm("golden") is Algorithm.GOLDEN_SECTION
    assert resolve_algorithm("simplex") is Algorithm.NELDER_MEAD
    assert resolve_algorithm("nelder mead") is Algorithm.NELDER_MEAD


def test_unknown_algorithm_lists_the_valid_algorithms():
    with pytest.raises(SolverError) as excinfo:
        resolve_algorithm("newton")  # not an engine this build has
    message = excinfo.value.message
    assert "Unknown algorithm" in message
    assert "golden-section-search" in message and "nelder-mead" in message


# --- the Nelder-Mead engine (33.14) -------------------------------------------


def test_nelder_mead_finds_a_two_variable_minimum():
    program = _annotated(
        "find-minimum of a 2-var paraboloid over x in [0, 5], y in [-4, 2]\n"
        "single minimum at (3, -1), value 0; the simplex walks both axes downhill",
        "(x - 3)**2 + (y + 1)**2",
    )
    result = nelder_mead(
        parse(program),
        [("x", 0.0, 5.0), ("y", -4.0, 2.0)],
        Mode.FLOATING_POINT,
        0,
        Objective.FIND_MINIMUM,
    )
    found = dict((name, value.to_float()) for name, value in result.solutions)
    assert found["x"] == pytest.approx(3.0, abs=1e-3)
    assert found["y"] == pytest.approx(-1.0, abs=1e-3)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-6)
    assert result.objective is Objective.FIND_MINIMUM
    assert result.algorithm == "nelder-mead"
    assert result.iterations > 0


def test_nelder_mead_solves_a_single_variable_root():
    program = _annotated(
        "find-root via Nelder-Mead (it minimises |expr|): x**2 - 2 over [0, 2]\n"
        "expect x = sqrt(2) ~ 1.41421",
        "x**2 - 2",
    )
    result = nelder_mead(
        parse(program), [("x", 0.0, 2.0)], Mode.FLOATING_POINT, 0, Objective.FIND_ROOT
    )
    assert result.solution.to_float() == pytest.approx(2**0.5, abs=1e-6)
    assert result.value.to_float() == pytest.approx(0.0, abs=1e-6)
    # The single-unknown convenience: solutions carries the one (name, value) pair.
    assert [name for name, _ in result.solutions] == ["x"]


def test_nelder_mead_finds_a_two_variable_maximum():
    program = _annotated(
        "find-maximum of a 2-var dome over x in [-2, 4], y in [-5, 1]\n"
        "peaks at (1, -2) with value 5",
        "5 - (x - 1)**2 - (y + 2)**2",
    )
    result = nelder_mead(
        parse(program),
        [("x", -2.0, 4.0), ("y", -5.0, 1.0)],
        Mode.FLOATING_POINT,
        0,
        Objective.FIND_MAXIMUM,
    )
    found = dict((name, value.to_float()) for name, value in result.solutions)
    assert found["x"] == pytest.approx(1.0, abs=1e-3)
    assert found["y"] == pytest.approx(-2.0, abs=1e-3)
    assert result.value.to_float() == pytest.approx(5.0, abs=1e-6)
    assert result.objective is Objective.FIND_MAXIMUM


def test_nelder_mead_reports_no_solution_naming_the_point():
    program = _annotated(
        "find-root of x**2 + y**2 + 1 over x, y in [-1, 1]\n"
        "no solution: never zero; closest is 1 at the origin. The error\n"
        "names the full multivariate point it reached",
        "x**2 + y**2 + 1",
    )
    with pytest.raises(SolverError) as excinfo:
        nelder_mead(
            parse(program),
            [("x", -1.0, 1.0), ("y", -1.0, 1.0)],
            Mode.FLOATING_POINT,
            0,
            Objective.FIND_ROOT,
        )
    message = excinfo.value.message
    assert "No solution" in message and "x =" in message and "y =" in message


def test_nelder_mead_unset_constant_surfaces_as_an_eval_error():
    program = _annotated(
        "find-minimum of a * x + y over x, y in [0, 1]\n"
        "as with golden-section, `a` is neither an unknown nor assigned: it\n"
        "fails at every vertex and is a structural error, not a region to avoid",
        "a * x + y",
    )
    with pytest.raises(EvalError) as excinfo:
        nelder_mead(
            parse(program),
            [("x", 0.0, 1.0), ("y", 0.0, 1.0)],
            Mode.FLOATING_POINT,
            0,
            Objective.FIND_MINIMUM,
        )
    assert "undefined variable: a" in excinfo.value.message
