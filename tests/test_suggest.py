# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit coverage of the nearest-match "did you mean" helper (TODO 43.5).

The pure core behind the function / mode / algorithm suggestions: given a typo'd
word and the set of valid spellings, return a ready-to-append hint or nothing.
The integration at each error site is covered in test_parser / test_server /
test_solver; this pins the helper's own contract — when it suggests, when it
stays silent, and the exact fragment it returns.
"""

from mcp_abacus.suggest import did_you_mean

_FUNCS = ["sqrt", "sin", "cos", "asin", "acos", "log", "exp", "floor", "ceil"]


def test_close_typo_is_suggested():
    # A one- or two-edit slip clears the cutoff and names the nearest candidate.
    assert did_you_mean("sqr", _FUNCS) == " Did you mean 'sqrt'?"
    assert did_you_mean("arcsin", _FUNCS) == " Did you mean 'asin'?"


def test_unrelated_word_gets_no_suggestion():
    # Below the similarity cutoff -> empty string, so a caller can splice it blindly
    # without ever emitting a misleading hint.
    assert did_you_mean("xyzzy", _FUNCS) == ""
    assert did_you_mean("", _FUNCS) == ""


def test_nearest_of_several_candidates_wins():
    # "asine" is closer to "asin" than to "sin"; the single best match is returned.
    assert did_you_mean("asine", _FUNCS) == " Did you mean 'asin'?"


def test_fragment_is_appendable():
    # The contract: a non-empty hint leads with one space and ends with '?', so it
    # appends cleanly onto an "unknown ..." sentence.
    hint = did_you_mean("flor", _FUNCS)
    assert hint.startswith(" ") and hint.endswith("?")
    assert hint == " Did you mean 'floor'?"


def test_empty_candidates_is_silent():
    assert did_you_mean("anything", []) == ""
