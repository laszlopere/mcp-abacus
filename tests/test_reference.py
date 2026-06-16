# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the language-help section text (TODO 21): sourced from live code."""

from mcp_abacus.expr import reference
from mcp_abacus.expr.lexer import _BASE_PREFIXES
from mcp_abacus.expr.nodes import FUNCTION_ARITIES, FUNCTION_HELP, UNARY_OPS
from mcp_abacus.expr.parser import _BINDING_POWER, _POWER_OPS
from mcp_abacus.expr.value import MODE_HELP, Mode


def test_types_section_lists_every_live_mode_with_its_description():
    text = reference.render("types")
    for mode in Mode:
        assert mode.value in text
        assert MODE_HELP[mode] in text


def test_types_section_has_one_line_per_mode():
    lines = reference.render("types").splitlines()
    assert len(lines) == len(list(Mode))


def test_language_section_mentions_every_operator():
    text = reference.render("language")
    for op in {*_BINDING_POWER, *UNARY_OPS, *_POWER_OPS}:
        assert op in text


def test_language_section_mentions_every_base_prefix():
    text = reference.render("language")
    for prefix in _BASE_PREFIXES:
        assert prefix.lower() in text


def test_language_section_states_power_bitwise_and_fixed_point_notation():
    text = reference.render("language")
    assert "POWER" in text  # ** is power
    assert "BITWISE" in text and "XOR" in text  # ^ & | ~ are bitwise; ^ is XOR
    assert "@" in text  # the M@D fixed-point literal


def test_language_section_documents_variables_and_statements():
    # The variable/multi-statement grammar (TODO 30) is offered to the caller:
    # assignment, bare-name reference, and the newline-separated statement list.
    text = reference.render("language")
    assert "name = expr" in text  # assignment form
    assert "reads a" in text and "variable" in text  # a bare name is a reference
    assert "newlines" in text  # statements are newline-separated
    assert "LAST statement" in text  # the program's value is the last statement's


def test_functions_section_mentions_every_function():
    text = reference.render("functions")
    for name in FUNCTION_ARITIES:
        assert f"{name}(" in text


def test_functions_section_marks_variadic_functions():
    text = reference.render("functions")
    for name, (_lo, hi) in FUNCTION_ARITIES.items():
        if hi is None:  # variadic — its signature must show the "one or more" tail
            assert f"{name}(" in text and "…" in text


def test_functions_section_gives_every_function_a_one_liner():
    # The drift guard: each registered function (incl. nullaries and the ln alias)
    # must have a FUNCTION_HELP entry, and that text must reach the rendered section.
    text = reference.render("functions")
    for name in FUNCTION_ARITIES:
        assert name in FUNCTION_HELP
        assert FUNCTION_HELP[name] in text


def test_function_help_has_no_entries_beyond_the_registry():
    # The mirror guard: no stale help for a function that was removed from the
    # registry — FUNCTION_HELP and FUNCTION_ARITIES cover exactly the same names.
    assert set(FUNCTION_HELP) == set(FUNCTION_ARITIES)


def test_solver_section_lists_every_live_objective_and_alias():
    # Sourced from the solver's own enum/aliases, so the help cannot drift from what
    # the tool accepts — every canonical objective and every alias must appear.
    from mcp_abacus.solver import _OBJECTIVE_ALIASES, Objective

    text = reference.render("solver")
    for objective in Objective:
        assert objective.value in text
    for alias in _OBJECTIVE_ALIASES:
        assert alias in text


def test_solver_section_states_the_bracket_and_unknown_rules():
    text = reference.render("solver")
    assert "bracket" in text
    assert "lower must be below upper" in text
    assert "must NOT" in text and "assigned" in text  # the unknown is free, not assigned


def test_unknown_section_lists_the_valid_sections_instead_of_erroring():
    text = reference.render("bogus")
    assert "bogus" in text
    assert "types" in text
    assert "language" in text
    assert "functions" in text
    assert "solver" in text


def test_every_mode_has_a_help_line():
    # The single-source guard: no Mode may ship without a MODE_HELP description.
    assert set(MODE_HELP) == set(Mode)
