# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end tests for vector literals over the real MCP stdio path (TODO 19.1.10).

Vectors are a NEW one-dimensional container held by the universal Value class. This
first cut covers construction + rendering only: a ``[a, b, …]`` literal builds a
vector in whatever scalar mode the calculation runs in, the result renders as
``[…]`` with its own exact/inexact verdict, and every scalar operator and function
REFUSES a vector with a clean message. There is no indexing and no arithmetic yet,
and nesting is rejected — those refusals are asserted here too.

Kept in its own file (not folded into test_e2e.py): the suite launches mcp-abacus as
a real child process and speaks MCP over stdio, the exact path a production client
takes. Helpers mirror test_e2e.py's; each test gets a fresh server subprocess.
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "mcp_abacus"])


@asynccontextmanager
async def _client():
    """Spawn the server over stdio and yield an initialized client session."""
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _payloads(calls):
    """Run a list of (tool, arguments) against ONE server; return parsed dicts in order."""

    async def go():
        async with _client() as session:
            out = []
            for tool, arguments in calls:
                result = await session.call_tool(tool, arguments)
                out.append(json.loads(result.content[0].text))
            return out

    return asyncio.run(go())


def _run_calc(cases):
    """Evaluate (expression, mode) cases via `calculate` against one server."""
    calls = []
    for expression, mode in cases:
        arguments = {"expression": expression}
        if mode is not None:
            arguments["mode"] = mode
        calls.append(("calculate", arguments))
    return _payloads(calls)


def _bare(payload):
    """The rendered value with its trailing ``(verdict)`` annotation stripped (25.1)."""
    return payload["value"].split(" (")[0]


def test_vector_literals_construct_and_render_across_modes():
    # A vector is built in the chosen scalar mode and renders as [a, b, …], each
    # element in that mode's own format; it is EXACT iff every element is exact.
    cases = [
        # (expression, mode, rendered, exact)
        ("[1, 2, 3]", "fixed-point", "[1, 2, 3]", True),
        ("[1.5, 2.5]", "fixed-point", "[1.5, 2.5]", True),  # each element keeps its scale-1
        ("[]", "fixed-point", "[]", True),  # empty is allowed (vacuously exact)
        ("[1+1, 2*3]", "fixed-point", "[2, 6]", True),  # elements are full expressions
        ("[1/3, 2/3]", "rational", "[1/3, 2/3]", True),  # rational holds them exactly
        ("[1.0, 2.0]", "floating-point", "[1.0, 2.0]", False),  # float is always inexact
    ]
    payloads = _run_calc([(e, m) for e, m, _r, _x in cases])
    for (expr, _mode, rendered, exact), payload in zip(cases, payloads, strict=True):
        assert payload["error"] is None, f"{expr!r} errored: {payload['error']}"
        assert _bare(payload) == rendered, f"{expr!r} -> {payload['value']!r}"
        assert payload["exact"] is exact, f"{expr!r} exact mismatch"
        assert payload["precision"] is None  # a vector has no single decimal scale
        assert payload["value_hex_dump"] is None  # nor a single integer to dump


def test_vector_with_an_inexact_element_is_inexact():
    # 1/3 does not fit fixed-point scale 0: it rounds to 0 and the element is inexact,
    # so the whole vector reports inexact (its verdict folds the elements' exactness).
    (payload,) = _run_calc([("[1/3]", "fixed-point")])
    assert payload["error"] is None
    assert _bare(payload) == "[0]"
    assert payload["exact"] is False


def test_vector_newline_is_insignificant_inside_brackets():
    # Like parens and argument lists, a NEWLINE inside [ ] is whitespace, so a vector
    # may span lines (30.5) — it does not split into two statements.
    (payload,) = _run_calc([("[1,\n2,\n3]", "fixed-point")])
    assert payload["error"] is None
    assert _bare(payload) == "[1, 2, 3]"


def test_vector_assigned_to_a_variable_and_read_back():
    # `name = [a, b, …]` binds a vector like any value (30.3); a later line reads it
    # back. The program's result is the last statement — the bare reference here.
    (payload,) = _run_calc([("v = [1.00, 1.50, 2.00]\nv", "fixed-point")])
    assert payload["error"] is None
    assert _bare(payload) == "[1.00, 1.50, 2.00]"
    assert payload["exact"] is True
    # The leading binding runs for effect; the result-bearing read line `v` shows the
    # vector it read back out of the run's store.
    (read,) = payload["values"]
    assert read["source"] == "v"
    assert read["value"].split(" (")[0] == "[1.00, 1.50, 2.00]"


def test_vector_assignment_alone_yields_the_vector():
    # A bare assignment also carries its own value (30.3), so a one-line program that
    # only binds still reports the vector as its result.
    (payload,) = _run_calc([("v = [1, 2, 3]", "fixed-point")])
    assert payload["error"] is None
    assert _bare(payload) == "[1, 2, 3]"


def test_operators_refuse_vectors():
    # No arithmetic on vectors yet (19.1.10): binary and unary operators refuse one
    # with a clean, self-contained diagnostic (value is None on the error path).
    cases = [
        ("[1,2] + [3,4]", "vectors do not support +"),
        ("[1,2] * [3,4]", "vectors do not support *"),
        ("-[1,2]", "vectors do not support unary -"),
        ("~[1,2]", "vectors do not support bitwise ~"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _ in cases])
    for (expr, message), payload in zip(cases, payloads, strict=True):
        assert payload["value"] is None, f"{expr!r} unexpectedly succeeded"
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"


def test_functions_refuse_a_vector_argument():
    # No function consumes a vector yet: passing one is refused uniformly, naming the
    # function. (Variadic stats like avg keep their loose-arg form, avg(1,2,3).)
    cases = [
        ("sqrt([1,2,3])", "sqrt() does not accept a vector"),
        ("avg([1,2,3])", "avg() does not accept a vector"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _ in cases])
    for (expr, message), payload in zip(cases, payloads, strict=True):
        assert payload["value"] is None, f"{expr!r} unexpectedly succeeded"
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"


def test_nested_vectors_are_rejected():
    # Strictly one-dimensional (19.1.10): an element that is itself a vector is refused.
    (payload,) = _run_calc([("[[1,2],[3,4]]", "fixed-point")])
    assert payload["value"] is None
    assert payload["error"] == "vectors must be one-dimensional; a vector cannot hold a vector"


def test_vector_is_not_a_selectable_mode():
    # VECTOR is an internal container, never a calculation mode: mode="vector" is
    # rejected exactly like an unknown name, and the valid-modes list never offers it.
    (payload,) = _run_calc([("[1,2,3]", "vector")])
    assert payload["value"] is None
    assert "Unknown mode: 'vector'" in payload["error"]
    assert "vector" not in payload["error"].split("Valid modes:")[1]


def test_analyze_renders_the_vector_node_and_its_scalar_children():
    # The analyze tree shows the VECTOR node with its element count and value, and each
    # element as its own scalar LITERAL leaf — proving the node walk descends into them.
    (payload,) = _payloads([("analyze", {"expression": "[1, 2]", "mode": "fixed-point"})])
    assert payload["error"] is None
    lines = payload["tree"].splitlines()
    assert lines[0] == "VECTOR (2 elements) Value = [1, 2] (vector, exact)"
    assert lines[1].lstrip().startswith('LITERAL "1" Value = 1 (fixed-point[0], exact)')
    assert lines[2].lstrip().startswith('LITERAL "2" Value = 2 (fixed-point[0], exact)')
