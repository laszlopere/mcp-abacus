# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the language-help section text (TODO 21): sourced from live code."""

from typing import get_args

from mcp_abacus.expr import reference
from mcp_abacus.expr.lexer import _BASE_PREFIXES
from mcp_abacus.expr.nodes import FUNCTION_ARITIES, FUNCTION_HELP, UNARY_OPS
from mcp_abacus.expr.parser import _BINDING_POWER, _POWER_OPS
from mcp_abacus.expr.value import MODE_ALIASES, MODE_HELP, Mode, selectable_modes


def test_types_section_lists_every_selectable_mode_with_its_description():
    # Only the SELECTABLE modes are advertised — the internal VECTOR container (19.1.10)
    # is deliberately omitted, since a caller can never pass it as `mode`.
    text = reference.render("types")
    for mode in selectable_modes():
        assert mode.value in text
        assert MODE_HELP[mode] in text
    assert Mode.VECTOR.value not in text


def test_types_section_has_one_line_per_selectable_mode():
    lines = reference.render("types").splitlines()
    assert len(lines) == len(selectable_modes())


def test_types_section_advertises_every_accepted_mode_alias():
    # TODO 41.11: every input-only alias resolve_mode honours must be documented, so a
    # caller can never be silently surprised by one (the `decimal` -> fixed-point trap).
    # Sourced from the live MODE_ALIASES map, so the help cannot drift from resolve_mode.
    text = reference.render("types")
    for alias in MODE_ALIASES:
        assert alias in text, f"types help omits the {alias!r} alias"


def test_types_section_lists_each_alias_on_its_target_modes_line():
    # The alias must sit on the line for the mode it resolves TO — `decimal` under
    # fixed-point, not floating-point — else the doc would mislead as badly as silence.
    lines = {line.split(" — ", 1)[0]: line for line in reference.render("types").splitlines()}
    for alias, mode in MODE_ALIASES.items():
        assert alias in lines[mode.value], f"{alias!r} not on the {mode.value!r} line"


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


def test_search_filter_keeps_only_matching_lines_case_insensitively():
    # The whole point: 'sin' over functions returns the sin family and nothing else.
    filtered = reference.render("functions", "SIN").splitlines()
    assert filtered  # non-empty
    assert all("sin" in line.lower() for line in filtered)
    names = {line.split("(", 1)[0].strip() for line in filtered}
    assert {"sin", "asin", "asinh", "sinh"} <= names


def test_search_filter_works_on_any_section_not_just_functions():
    filtered = reference.render("types", "rational")
    assert filtered  # the rational mode line survives
    assert all("rational" in line.lower() for line in filtered.splitlines())


def test_empty_or_omitted_search_filter_returns_the_whole_section():
    full = reference.render("functions")
    assert reference.render("functions", None) == full
    assert reference.render("functions", "") == full


def test_search_filter_that_matches_nothing_reports_instead_of_emptiness():
    text = reference.render("functions", "no_such_function_xyz")
    assert "no_such_function_xyz" in text
    assert "match" in text


# A detail card's header is the only line at column 0; the prose under it is indented.
def _detail_headers(text):
    return [ln for ln in text.splitlines() if ln and not ln[0].isspace()]


def test_details_renders_a_card_with_signature_arity_and_description():
    from mcp_abacus.expr.function_details import FUNCTION_DETAILS
    from mcp_abacus.expr.nodes import _describe_arity

    out = reference.render("functions", "atan2", details=True)
    header = next(h for h in _detail_headers(out) if h.startswith("atan2("))
    assert _describe_arity(*FUNCTION_ARITIES["atan2"]) in header  # arity on the header
    # the long-form prose (not the one-liner) is what the card carries
    assert FUNCTION_DETAILS["atan2"].split("\n\n")[0] in out
    assert FUNCTION_HELP["atan2"] not in out  # the terse one-liner is NOT the detail


def test_details_selects_the_same_functions_as_the_plain_filter():
    # The detail set must mirror the row filter exactly — same substring, same names.
    detailed = {
        h.split("(", 1)[0]
        for h in _detail_headers(reference.render("functions", "sin", details=True))
    }
    plain = {
        ln.split("(", 1)[0].strip() for ln in reference.render("functions", "sin").splitlines()
    }
    assert detailed == plain
    assert {"sin", "asin", "asinh", "sinh"} <= detailed


def test_details_with_no_filter_cards_every_function():
    headers = _detail_headers(reference.render("functions", details=True))
    assert len(headers) == len(FUNCTION_ARITIES)


def test_function_details_covers_exactly_the_registry():
    # The drift guard: every function has long-form detail text and there is no stale
    # entry for a removed function — FUNCTION_DETAILS and FUNCTION_ARITIES share names.
    from mcp_abacus.expr.function_details import FUNCTION_DETAILS

    assert set(FUNCTION_DETAILS) == set(FUNCTION_ARITIES)


def test_each_function_detail_is_nonempty_and_at_most_three_paragraphs():
    # The authoring contract: prose present, blank-line-separated, capped at 3 paras.
    from mcp_abacus.expr.function_details import FUNCTION_DETAILS

    for name, text in FUNCTION_DETAILS.items():
        assert text.strip(), f"{name} has empty detail text"
        paragraphs = text.split("\n\n")
        assert len(paragraphs) <= 3, f"{name} has {len(paragraphs)} paragraphs (>3)"
        assert all(p.strip() for p in paragraphs), f"{name} has a blank paragraph"


def test_details_is_a_no_op_on_non_function_sections():
    # The other sections carry no per-item detail, so details must not alter them.
    assert reference.render("types", details=True) == reference.render("types")
    assert reference.render("solver", details=True) == reference.render("solver")


def test_details_that_matches_no_function_reports_instead_of_emptiness():
    text = reference.render("functions", "no_such_function_xyz", details=True)
    assert "no_such_function_xyz" in text
    assert "match" in text


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


def test_index_and_sections_list_every_registered_section():
    # TODO 41.8: the resource index is built from the same _SECTIONS registry, so it
    # must name every section and stay in step with sections().
    assert set(reference.sections()) == set(reference._SECTIONS)
    index = reference.index()
    for name, (desc, _builder) in reference._SECTIONS.items():
        assert name in index
        assert desc in index


def test_help_section_enum_mirrors_the_reference_sections():
    # TODO 41.2: the help tool's `section` is a Literal enum (server.HelpSection) so
    # clients see the valid values in the schema. It must stay in lockstep with the
    # actual _SECTIONS registry — a section added/removed in reference.py without
    # updating the Literal (or vice versa) is caught here.
    from mcp_abacus.server import HelpSection

    assert set(get_args(HelpSection)) == set(reference._SECTIONS)
