# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end tests for vector literals over the real MCP stdio path (TODO 19.1.10).

Vectors are a NEW one-dimensional container held by the universal Value class. This
first cut covers construction + rendering: a ``[a, b, …]`` literal builds a vector
in whatever scalar mode the calculation runs in, and the result renders as ``[…]``
with its own exact/inexact verdict. The stats family reduces over a vector's elements
— ``max``/``min``/``avg``/``median``/``variance``/``stddev`` (28.2–28.9) — and
``covariance`` takes two vectors (40.13); every OTHER scalar operator and function
still REFUSES a vector with a clean message. Going the other way, ``factor`` (40.23)
is the first vector-PRODUCING function: a scalar integer in, a vector of its prime
factors out. There is no indexing and no general arithmetic yet, and nesting is
rejected — those refusals are asserted here too.

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
    # Most functions don't consume a vector: passing one is refused uniformly, naming
    # the function. (A variadic arity is no protection — factorial-style integer
    # functions still refuse.) The exceptions are the reducing family (max/min/avg/
    # median/variance/stddev and the integer reducers gcd/lcm take a single vector,
    # covariance/correlation take two) — see the next tests.
    cases = [
        ("sqrt([1,2,3])", "sqrt() does not accept a vector"),
        ("factorial([1,2,3])", "factorial() does not accept a vector"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _ in cases])
    for (expr, message), payload in zip(cases, payloads, strict=True):
        assert payload["value"] is None, f"{expr!r} unexpectedly succeeded"
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"


def test_max_and_min_reduce_over_a_vector():
    # max/min are the first vector-CONSUMING functions (28.2/28.3): a single vector
    # operand is reduced over its elements, selecting one VERBATIM (its own scale). A
    # vector mixed with anything else, or an empty one, refuses over the real stdio path.
    cases = [
        ("max([3, 1, 2])", "3 (exact)", None),
        ("min([3, 1, 2])", "1 (exact)", None),
        ("min([1.50, 1.5])", "1.50 (exact)", None),  # tie -> earliest, its scale wins
        (
            "min([1, 2], 3)",
            None,
            "min has two forms — min(vector) or min(a, b, …) — and cannot mix them",
        ),
        ("max([])", None, "max of an empty vector is undefined"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _, _ in cases])
    for (expr, value, message), payload in zip(cases, payloads, strict=True):
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"
        assert payload["value"] == value, f"{expr!r} -> {payload['value']!r}"


def test_computing_stats_reduce_over_a_vector():
    # avg/median/variance/stddev join max/min as vector-CONSUMING (28.4/28.7/28.8/28.9):
    # a single vector is reduced over its elements, same as the equivalent flat run.
    cases = [
        ("avg([2, 4, 6])", "4 (exact)", None),
        ("median([3, 1, 2])", "2 (exact)", None),
        ("variance([1, 2, 3, 4, 5])", "2 (exact)", None),
        ("stddev([2, 4, 4, 4, 5, 5, 7, 9])", "2 (exact)", None),
        ("variance([])", None, "variance of an empty vector is undefined"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _, _ in cases])
    for (expr, value, message), payload in zip(cases, payloads, strict=True):
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"
        assert payload["value"] == value, f"{expr!r} -> {payload['value']!r}"


def test_integer_reducers_reduce_over_a_vector():
    # gcd/lcm (40.7/40.8) join the reducing family: a single vector is reduced over its
    # elements, same as the equivalent flat run, keeping the integer-only domain. A
    # vector mixed with a scalar, or an empty one, refuses; a fractional element still
    # hits the integer-domain gate.
    cases = [
        ("gcd([54, 24, 6])", "6 (exact)", None),
        ("lcm([2, 3, 4])", "12 (exact)", None),
        ("gcd([42])", "42 (exact)", None),  # single-element vector folds to |a|
        (
            "gcd([1, 2], 3)",
            None,
            "gcd has two forms — gcd(vector) or gcd(a, b, …) — and cannot mix them",
        ),
        ("lcm([])", None, "lcm of an empty vector is undefined"),
        ("gcd([2.5, 5])", None, "gcd requires integer operands"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _, _ in cases])
    for (expr, value, message), payload in zip(cases, payloads, strict=True):
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"
        assert payload["value"] == value, f"{expr!r} -> {payload['value']!r}"


def test_covariance_takes_two_vectors():
    # covariance (40.13) is the first TWO-vector function: paired population covariance
    # mean((x-mx)*(y-my)), refusing unequal-length, empty, or non-vector arguments.
    cases = [
        ("covariance([2, 4, 6, 8], [1, 3, 5, 7])", "5 (exact)", None),
        (
            "covariance([1, 2, 3], [6, 5, 4])",
            "-2/3 (exact)",
            None,
        ),  # anti-correlated, exact rational
        ("covariance([1, 2, 3], [4, 5])", None, "covariance requires two equal-length vectors"),
        ("covariance([1, 2], 3)", None, "covariance takes two vectors — covariance(x, y)"),
    ]
    payloads = _run_calc([(e, "rational") for e, _, _ in cases])
    for (expr, value, message), payload in zip(cases, payloads, strict=True):
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"
        assert payload["value"] == value, f"{expr!r} -> {payload['value']!r}"


def test_correlation_takes_two_vectors():
    # correlation (40.14) is the second TWO-vector function: Pearson r in [-1, 1],
    # cov/(stddev*stddev). A perfectly (anti-)correlated pair is +1 / -1; a constant
    # series has zero stddev and divides by zero; the domain refusals name correlation.
    cases = [
        ("correlation([1, 2, 3], [2, 4, 6])", "1.0 (inexact)", None),
        ("correlation([1, 2, 3], [6, 4, 2])", "-1.0 (inexact)", None),
        ("correlation([2, 2, 2], [1, 2, 3])", None, "float division by zero"),
        ("correlation([1, 2, 3], [4, 5])", None, "correlation requires two equal-length vectors"),
    ]
    payloads = _run_calc([(e, "floating-point") for e, _, _ in cases])
    for (expr, value, message), payload in zip(cases, payloads, strict=True):
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"
        assert payload["value"] == value, f"{expr!r} -> {payload['value']!r}"


def test_factor_produces_a_vector_of_primes():
    # factor (40.23) is the first vector-PRODUCING function: a scalar integer in, a
    # VECTOR of its prime factors out — ascending, WITH multiplicity, so the product
    # of the elements reconstructs n. A prime factors to itself; factor(1) is the
    # empty product []. Default fixed-point mode renders each factor at scale 0.
    cases = [
        # (expression, rendered, exact)
        ("factor(12)", "[2, 2, 3]", True),
        ("factor(7)", "[7]", True),  # a prime factors to itself
        ("factor(1)", "[]", True),  # the empty product, vacuously exact
        ("factor(360)", "[2, 2, 2, 3, 3, 5]", True),
        ("factor(97)", "[97]", True),  # a larger prime, found above the wheel
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _r, _x in cases])
    for (expr, rendered, exact), payload in zip(cases, payloads, strict=True):
        assert payload["error"] is None, f"{expr!r} errored: {payload['error']}"
        assert _bare(payload) == rendered, f"{expr!r} -> {payload['value']!r}"
        assert payload["exact"] is exact, f"{expr!r} exact mismatch"
        assert payload["precision"] is None  # a vector has no single decimal scale
        assert payload["value_hex_dump"] is None  # nor a single integer to dump


def test_factor_follows_the_run_mode_for_its_elements():
    # The produced vector carries the run's element mode, exactly as a vector LITERAL
    # would: rational holds the factors exactly, floating-point renders them as inexact
    # doubles (a float vector is always inexact). The empty product stays exact in
    # every mode.
    cases = [
        # (expression, mode, rendered, exact)
        ("factor(12)", "rational", "[2, 2, 3]", True),
        ("factor(12)", "floating-point", "[2.0, 2.0, 3.0]", False),
        ("factor(1)", "floating-point", "[]", True),
    ]
    payloads = _run_calc([(e, m) for e, m, _r, _x in cases])
    for (expr, _mode, rendered, exact), payload in zip(cases, payloads, strict=True):
        assert payload["error"] is None, f"{expr!r} errored: {payload['error']}"
        assert _bare(payload) == rendered, f"{expr!r} -> {payload['value']!r}"
        assert payload["exact"] is exact, f"{expr!r} exact mismatch"


def test_factor_refusals():
    # DOMAIN positive integers only. 0 and negatives refuse; a non-integer hits the
    # shared integer-domain gate (gcd/lcm, 40.7/40.8); a value past the cap refuses
    # rather than locking up trial division; and a vector argument is refused like any
    # scalar-only function — factor PRODUCES a vector, it does not consume one.
    cases = [
        ("factor(0)", "factor requires a positive integer"),
        ("factor(-12)", "factor requires a positive integer"),
        ("factor(2.5)", "factor requires integer operands"),
        ("factor(1000000000001)", "factor argument too large (limit 1000000000000)"),
        ("factor([2, 3])", "factor() does not accept a vector"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _ in cases])
    for (expr, message), payload in zip(cases, payloads, strict=True):
        assert payload["value"] is None, f"{expr!r} unexpectedly succeeded"
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"


def test_map_transforms_a_vector_element_wise():
    # map (40.24) is the first function that both CONSUMES and PRODUCES a vector: it evaluates
    # the body for each element with the 2nd-arg NAME bound to it, collecting a same-length
    # vector. It composes with factor (the other vector-producer) and reads outer bindings; a
    # scalar 1st arg or a non-name 2nd arg refuses over the real stdio path.
    cases = [
        ("map([1, 2, 3], x, x**2)", "[1, 4, 9]", True, None),
        ("map([], x, x**2)", "[]", True, None),  # empty maps to empty, vacuously exact
        ("map(factor(12), x, x**2)", "[4, 4, 9]", True, None),  # consumes a produced vector
        ("k = 10\nmap([1, 2, 3], x, k*x)", "[10, 20, 30]", True, None),  # reads an outer binding
        ("map(5, x, x**2)", None, None, "map's first argument must be a vector"),
        ("map([1, 2], 3, x)", None, None, "map's variable (2nd argument) must be a name"),
    ]
    payloads = _run_calc([(e, "fixed-point") for e, _r, _x, _m in cases])
    for (expr, rendered, exact, message), payload in zip(cases, payloads, strict=True):
        assert payload["error"] == message, f"{expr!r} -> {payload['error']!r}"
        if message is None:
            assert _bare(payload) == rendered, f"{expr!r} -> {payload['value']!r}"
            assert payload["exact"] is exact, f"{expr!r} exact mismatch"
            assert payload["precision"] is None  # a vector has no single decimal scale
        else:
            assert payload["value"] is None, f"{expr!r} unexpectedly succeeded"


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
