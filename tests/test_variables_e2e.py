# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end functional tests for variables (TODO 30) through the `calculate` tool.

The companion to test_variables.py's unit-level coverage: where that drives
parse()+evaluate() directly, this exercises the WHOLE seam — multi-line abacus
programs go in as the tool's `expression` argument and the annotated `value`
string (or the line-tagged error) comes back, the same path a real client takes
(dispatch -> parse -> evaluate -> render). The programs use the variable features
in concert: assignment, back-reference, reassignment, variables feeding functions
and nullaries, and the store's per-mode value passthrough.

Under `pytest -v` (run-tests.sh --verbose) conftest's _compact_variables_trace
prints each program's source lines and the result it produced.
"""

import asyncio
import json

import pytest

from mcp_abacus.server import mcp


def _calc(expression, mode=None, floor=None):
    """Invoke `calculate` in-process; return its structured payload dict (25.1).

    `mode` is omitted from the request when None, exercising the fixed-point default;
    `floor` (min_fixed_point_precision) is likewise omitted when None.
    """
    arguments = {"expression": expression}
    if mode is not None:
        arguments["mode"] = mode
    if floor is not None:
        arguments["min_fixed_point_precision"] = floor
    result = asyncio.run(mcp.call_tool("calculate", arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def _value(expression, mode=None, floor=None):
    """The annotated `value` string on success; fail loudly if the call errored."""
    payload = _calc(expression, mode, floor)
    assert payload["error"] is None, f"{expression!r} [{mode}] errored: {payload['error']}"
    return payload["value"]


# --- assignment yields its value, references read it back -------------------


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # A lone assignment is also an expression: its value is the bound RHS (30.6).
        ("x = 2 * 21", None, "42 (exact)"),
        # Bind then reference on the next line — the store carries x across the run.
        ("x = 2 + 3\nx", None, "5 (exact)"),
        # A reference threads through further arithmetic like any other atom.
        ("x = 41\nx + 1", None, "42 (exact)"),
        # One variable reused several times in the final expression.
        ("n = 5\nn * n * n", None, "125 (exact)"),
    ],
)
def test_assignment_and_reference(expression, mode, value):
    assert _value(expression, mode) == value


# --- multi-line programs: dependent variables, reassignment ----------------


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # A later variable depends on an earlier one; the last line is the result.
        ("x = 10\ny = x * 2\ny + 1", None, "21 (exact)"),
        # Reassignment overwrites in place — x = x + 5 reads the old x, binds the new.
        ("x = 1\nx = x + 5\nx", None, "6 (exact)"),
        # A money-shaped program: fixed-point keeps the covering scale exactly.
        ("price = 19.99\nqty = 3\nprice * qty", None, "59.97 (exact)"),
        ("principal = 1000.00\nrate = 5\nprincipal * rate / 100", None, "50.00 (exact)"),
        ("net = 100.00\nvat = 0.27\nnet + net * vat", None, "127.00 (exact)"),
        # Variables standing in for the parts of a bigger formula.
        ("base = 2\nexp = 10\nbase ** exp", None, "1024 (exact)"),
        ("r = 3\narea = r * r\narea", None, "9 (exact)"),
    ],
)
def test_multi_line_programs(expression, mode, value):
    assert _value(expression, mode) == value


# --- variables feeding functions and nullaries -----------------------------


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # A variable as a function argument: sqrt(16) bound, then used.
        ("s = sqrt(16)\ns + 1", None, "5 (exact)"),
        # Variables across a multi-argument call — the 3-4-5 right triangle.
        ("a = 3\nb = 4\nsqrt(a*a + b*b)", None, "5 (exact)"),
        # A nullary's value bound to a variable, then scaled (float, so inexact).
        ("c = pi()\n2 * c", "floating-point", "6.283185307179586 (inexact)"),
    ],
)
def test_variables_with_functions(expression, mode, value):
    assert _value(expression, mode) == value


# --- per-mode value passthrough --------------------------------------------
# The store holds Values verbatim — no mode or scale of its own — so a reference
# reproduces exactly what the assignment computed, in whatever mode the run uses.


@pytest.mark.parametrize(
    ("mode", "value"),
    [
        (None, "5 (exact)"),  # fixed-point default
        ("rational", "5 (exact)"),
        ("floating-point", "5.0 (inexact)"),  # float carries its inexact flag through
    ],
)
def test_passthrough_across_modes(mode, value):
    # The SAME program in each mode: the read-back x is the mode's own 2 + 3.
    assert _value("x = 2 + 3\nx", mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # rational keeps the exact fraction a reference was bound to.
        ("a = 1/2\nb = 1/3\na + b", "rational", "5/6 (exact)"),
        # float division read back through a variable stays inexact.
        ("x = 2\nx / 4", "floating-point", "0.5 (inexact)"),
        # An inexact (rounded) fixed-point sqrt is stored and read back UNCHANGED:
        # the store re-coerces nothing, so the inexact flag and offer hint survive.
        (
            "root = sqrt(2.000000)\nroot",
            None,
            "1.414214 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.4142135624)",
        ),
    ],
)
def test_passthrough_preserves_value_shape(expression, mode, value):
    assert _value(expression, mode) == value


def test_min_fixed_point_precision_threads_through_variables():
    # The precision floor reaches the literals inside an assignment's RHS, so a
    # variable holding 1/3 carries 5 decimals into the final d + d (25.2.1).
    bare = _value("d = 1 / 3\nd + d")
    assert bare.startswith("0 (inexact, rounded to 0 decimals")
    floored = _value("d = 1 / 3\nd + d", floor=5)
    assert floored == "0.66666 (inexact, rounded to 5 decimals)"


# --- undefined-variable refusal is a plain, self-contained message ----------
#
# The diagnostic names the missing variable and stands on its own — no machine-style
# "error (line N):" prefix (only the inexact-abort headline names a line, 35.3.1).


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("z + 1", "undefined variable: z"),  # a bare unset name
        ("x = 1\nz", "undefined variable: z"),  # an unset reference on a later line
        ("a = 5\nb = a + 1\nc", "undefined variable: c"),
    ],
)
def test_undefined_variable_refuses_with_a_plain_message(expression, error):
    payload = _calc(expression)
    assert payload["error"] == error
    assert payload["value"] is None


# --- pi/e are reserved constants: assigning to one is rejected (29.6) -------


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("pi = 3", "'pi' is a constant and cannot be assigned"),
        ("e = 2", "'e' is a constant and cannot be assigned"),
        # The guard fires on the assignment target even mid-program, before later lines.
        ("x = 1\npi = 4", "'pi' is a constant and cannot be assigned"),
    ],
)
def test_assigning_to_a_constant_refuses(expression, error):
    payload = _calc(expression)
    assert payload["error"] == error
    assert payload["value"] is None


def test_bare_constant_works_through_the_tool():
    # The benchmarks regression (29.6): `2*pi` once errored "undefined variable: pi";
    # now it resolves to the constant like any literal would.
    assert _value("2 * pi", "floating-point") == "6.283185307179586 (inexact)"


# --- every bare-expression line is answered, not just the last (TODO) -------


def test_all_output_expressions_returned():
    # The driving regression: a program with one assignment and two bare divisions —
    # both divisions must come back in `values`, in source order, and the silent
    # `x = 10` assignment must NOT (only bare lines, plus the always-echoed final).
    payload = _calc("x = 10\n(x - 1) / (x + 1)\n(x + 1) / 100.0")
    assert payload["error"] is None
    sources = [entry["source"] for entry in payload["values"]]
    assert sources == ["(x - 1) / (x + 1)", "(x + 1) / 100.0"]


def test_value_is_transcript_for_multiple_outputs():
    # With 2+ answered lines the top-level `value` is a `<expr> = <result>` transcript,
    # one line per `values` entry, in order.
    payload = _calc("x = 10\n(x - 1) / (x + 1)\n(x + 1) / 100.0")
    expected = "\n".join(f"{e['source']} = {e['value']}" for e in payload["values"])
    assert payload["value"] == expected
    assert payload["value"].count("\n") == 1  # exactly two lines


def test_values_last_entry_matches_top_level():
    # The last answered line IS the program's result, so its scalar fields equal the
    # top-level ones (the `value` string may differ — transcript vs bare).
    payload = _calc("a = 1 / 3\nb = 1 / 7\na\nb")
    last = payload["values"][-1]
    for key in ("value_hex_dump", "exact", "precision", "offered_precision"):
        assert last[key] == payload[key]


def test_single_output_unchanged():
    # A lone bare expression: one `values` entry, no transcript, `value` byte-identical
    # to the entry's value (full backward compatibility).
    payload = _calc("2 + 3")
    assert len(payload["values"]) == 1
    assert payload["value"] == "5 (exact)"
    assert payload["values"][0]["value"] == "5 (exact)"
    assert payload["values"][0]["source"] == "2 + 3"


def test_assignments_run_silently_final_line_echoed():
    # An all-assignment program: assignments are silent, but the final line is always
    # the result, so `values` holds exactly it — and there is no transcript prefix.
    payload = _calc("x = 1\ny = 2")
    assert [e["source"] for e in payload["values"]] == ["y = 2"]
    assert payload["value"] == "2 (exact)"


def test_per_statement_offered_precision():
    # Each answered line steers independently: the inexact `1 / 3` line carries its own
    # offered_precision, the exact `2 + 2` line offers nothing.
    payload = _calc("1 / 3\n2 + 2")
    inexact, exact = payload["values"]
    assert inexact["source"] == "1 / 3"
    assert inexact["offered_precision"] is not None
    assert exact["offered_precision"] is None


def test_per_statement_offer_rebuilds_bindings():
    # Regression guard for the snapshot/re-run ordering: a later bare line that reads an
    # earlier binding must get the RIGHT offered value, proving the whole-program re-run
    # at the higher floor rebuilt `d` rather than evaluating the line in isolation.
    payload = _calc("d = 1 / 3\nd + d\nd * 3")
    offered = payload["values"][0]["offered_precision"]
    # d + d at floor 4 = 0.3333 + 0.3333 = 0.6666 (had `d` been lost, this would error or
    # be wrong); the offer reveals those digits the scale-0 result (1) hid.
    assert offered["value"].startswith("0.6666")


def test_floating_point_outputs_have_null_offers():
    # offered_precision is fixed-point-only: every float line offers nothing, precision null.
    payload = _calc("1 / 3\n2 / 7", mode="floating-point")
    for entry in payload["values"]:
        assert entry["offered_precision"] is None
        assert entry["precision"] is None
