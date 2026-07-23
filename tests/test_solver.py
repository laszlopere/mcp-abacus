# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit-level coverage of the solver module's pure helpers (TODO 31 / 32 / 33.14).

The companion to test_solver_e2e.py: where that drives the whole `solver` TOOL
seam (request dict -> dispatch -> search -> reply) and shows each round-trip, this
exercises the building blocks in isolation — objective/algorithm resolution, the
objective fold, and the bracket / unknown validators — without running a search.
"""

import pytest

from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import Mode, Value
from mcp_abacus.solver import (
    Algorithm,
    Objective,
    SolverError,
    fold_objective,
    resolve_algorithm,
    resolve_objective,
    validate_bracket,
    validate_unknown,
)


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


# --- algorithm resolution (33.14) ---------------------------------------------


def test_algorithm_defaults_to_golden_section_when_omitted():
    assert resolve_algorithm(None) is Algorithm.GOLDEN_SECTION


def test_canonical_algorithms_resolve():
    assert resolve_algorithm("golden-section-search") is Algorithm.GOLDEN_SECTION
    assert resolve_algorithm("brent-parabolic") is Algorithm.BRENT_PARABOLIC
    assert resolve_algorithm("brent-dekker") is Algorithm.BRENT_DEKKER
    assert resolve_algorithm("chandrupatla") is Algorithm.CHANDRUPATLA
    assert resolve_algorithm("secant") is Algorithm.SECANT
    assert resolve_algorithm("newton-raphson") is Algorithm.NEWTON_RAPHSON
    assert resolve_algorithm("halley") is Algorithm.HALLEY
    assert resolve_algorithm("nelder-mead") is Algorithm.NELDER_MEAD


def test_algorithm_aliases_resolve():
    # Never-surfaced spellings still resolve to the canonical engine.
    assert resolve_algorithm("golden") is Algorithm.GOLDEN_SECTION
    assert resolve_algorithm("brent") is Algorithm.BRENT_PARABOLIC
    assert resolve_algorithm("parabolic") is Algorithm.BRENT_PARABOLIC
    assert resolve_algorithm("simplex") is Algorithm.NELDER_MEAD
    assert resolve_algorithm("nelder mead") is Algorithm.NELDER_MEAD


def test_the_two_brents_stay_distinct():
    # 33.2: the root finder is spelled out as brent-dekker, and bare `brent` KEEPS naming
    # the parabolic minimiser it always named — so no pre-33.2 call changes meaning. The
    # root method takes the spellings a caller would otherwise reach for with `brent`.
    assert resolve_algorithm("brent") is Algorithm.BRENT_PARABOLIC
    assert resolve_algorithm("brent-dekker") is Algorithm.BRENT_DEKKER
    assert resolve_algorithm("brent-root") is Algorithm.BRENT_DEKKER
    assert resolve_algorithm("brent-method") is Algorithm.BRENT_DEKKER
    assert resolve_algorithm("dekker") is Algorithm.BRENT_DEKKER
    assert resolve_algorithm("zbrent") is Algorithm.BRENT_DEKKER


def test_chandrupatla_aliases_resolve():
    # 33.7: the possessive and -method spellings of an awkward-to-type name.
    assert resolve_algorithm("chandrupatlas") is Algorithm.CHANDRUPATLA
    assert resolve_algorithm("chandrupatla-method") is Algorithm.CHANDRUPATLA


def test_secant_aliases_resolve():
    # 33.3: the -method spelling, and `chord` — the other textbook name for the same step.
    assert resolve_algorithm("secant-method") is Algorithm.SECANT
    assert resolve_algorithm("chord") is Algorithm.SECANT


def test_newton_aliases_resolve():
    # 33.4: bare `newton` names the ROOT finder — the same first-come rule bare `brent`
    # follows for the parabolic minimiser — plus the possessive / -method / spaced forms.
    assert resolve_algorithm("newton") is Algorithm.NEWTON_RAPHSON
    assert resolve_algorithm("newton-method") is Algorithm.NEWTON_RAPHSON
    assert resolve_algorithm("newtons-method") is Algorithm.NEWTON_RAPHSON
    assert resolve_algorithm("newton raphson") is Algorithm.NEWTON_RAPHSON
    assert resolve_algorithm("raphson") is Algorithm.NEWTON_RAPHSON


def test_halley_aliases_resolve():
    # 33.8: the possessive and -method spellings, as for the other named methods.
    assert resolve_algorithm("halleys") is Algorithm.HALLEY
    assert resolve_algorithm("halley-method") is Algorithm.HALLEY
    assert resolve_algorithm("halleys-method") is Algorithm.HALLEY


def test_unknown_algorithm_lists_the_valid_algorithms():
    with pytest.raises(SolverError) as excinfo:
        resolve_algorithm("householder")  # not an engine this build has
    message = excinfo.value.message
    assert "Unknown algorithm" in message
    assert "golden-section-search" in message and "nelder-mead" in message


def test_unknown_algorithm_suggests_the_nearest_spelling():
    # 43.5: a near-miss spelling gets a "did you mean" pointing at the closest engine
    # (canonical names or aliases), so a typo self-corrects in one turn.
    with pytest.raises(SolverError) as excinfo:
        resolve_algorithm("bisecton")  # one letter off "bisection"
    assert "Did you mean 'bisection'?" in excinfo.value.message


def test_unknown_algorithm_far_miss_gets_no_suggestion():
    # An unrelated word clears no candidate above the cutoff, so NO suggestion is made —
    # a wrong "did you mean" would be worse than none. Only the valid list is offered.
    with pytest.raises(SolverError) as excinfo:
        resolve_algorithm("zzz")
    message = excinfo.value.message
    assert "Did you mean" not in message
    assert "Valid algorithms" in message
