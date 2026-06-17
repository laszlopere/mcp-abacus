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
