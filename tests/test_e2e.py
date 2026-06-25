# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end tests for the `calculate` tool (TODO 23).

test_server.py drives the FastMCP app object in-process. These tests go one
layer further out: they launch mcp-abacus as a real child process and speak the
MCP protocol to it over stdio — the exact path a production client (Claude)
takes. So they prove the tool survives the full initialize -> list_tools ->
call_tool round-trip, that the default and explicit `mode` arguments travel
across the wire intact, and that the error / unknown-mode paths come back as
readable text content rather than a protocol-level failure.

Each test opens its own server subprocess for isolation; a round-trip is a
fraction of a second, so the small suite stays fast. No pytest-asyncio
dependency — we drive the async client with asyncio.run, as test_server.py does.
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

# Launch the server with the interpreter already running the tests (the venv
# that has mcp_abacus installed), via its console module entry point.
SERVER = StdioServerParameters(command=sys.executable, args=["-m", "mcp_abacus"])


@asynccontextmanager
async def _client():
    """Spawn the server over stdio and yield an initialized client session."""
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _call(tool, arguments=None):
    """Synchronously invoke one tool against a fresh server; return the result."""

    async def go():
        async with _client() as session:
            return await session.call_tool(tool, arguments or {})

    return asyncio.run(go())


def _text(result):
    """The single text block of a tool result."""
    return result.content[0].text


def _payload(result):
    """calculate's structured dict (25.1), rendered by FastMCP as JSON text."""
    return json.loads(_text(result))


def _value(result):
    """The bare rendered number (precision annotation stripped) on success.

    calculate now annotates `value` with its precision verdict (25.1); these
    arithmetic-over-the-wire tests assert the number, so strip the "(exact)" /
    "(inexact, ...)" tail here. The verdict itself is asserted by its own tests.
    """
    payload = _payload(result)
    return payload["value"].split(" (")[0]


def _calc_in_one_session(calls):
    """Evaluate (expression, mode) pairs against ONE server; return their texts.

    Keeps the multi-call tests on a single subprocess (mode is omitted from the
    request when None, exercising the default). Order is preserved.
    """

    async def go():
        async with _client() as session:
            texts = []
            for expression, mode in calls:
                arguments = {"expression": expression}
                if mode is not None:
                    arguments["mode"] = mode
                result = await session.call_tool("calculate", arguments)
                texts.append(_value(result))
            return texts

    return asyncio.run(go())


def _analyze_in_one_session(calls):
    """Invoke `analyze` for each (expression, mode, floor) against ONE server.

    Returns the list of structured payloads in order. `mode`/`min_fixed_point_
    precision` are omitted from the request when None, so the defaults are
    exercised on the wire.
    """

    async def go():
        async with _client() as session:
            payloads = []
            for expression, mode, floor in calls:
                arguments = {"expression": expression}
                if mode is not None:
                    arguments["mode"] = mode
                if floor is not None:
                    arguments["min_fixed_point_precision"] = floor
                result = await session.call_tool("analyze", arguments)
                payloads.append((arguments, _payload(result)))
            return payloads

    return asyncio.run(go())


def _tracing(request):
    """True only under ./run-tests.sh --verbose / --human-readable.

    The `analyze` tree-printing below predates the autouse `_compact_e2e_trace`
    (which traces `calculate`, not `analyze`), so it prints inline — but gated on
    the same switches as the rest of the suite, mirroring conftest's
    ``_verbose_trace``. A plain run stays silent.
    """
    return request.config.option.verbose >= 1 or request.config.option.human_readable


def test_analyze_renders_evaluated_trees_across_modes_and_precision_over_the_wire(capsys, request):
    # End-to-end over real stdio: each case is (expression, mode, floor) with the
    # tree ROOT it must evaluate to. The cases span all three modes and the fixed-
    # point floor, and each is chosen so the tree EXPLAINS its root:
    #   - (1 + 1/2)*3 is 3 in fixed-point (the 1/2 -> 0 leaf rounds the half away),
    #     4.5 in floating-point, and 4.50 once the floor keeps two decimals;
    #   - (0.1 + 0.2)*3 is exactly 9/10 in rational;
    #   - 1 ETH in wei (hex mantissa @18) quartered stays exact in fixed-point;
    #   - mixed power/rational and floor-div/modulo round-trip as whole numbers.
    cases = [
        ("(1 + 1/2) * 3", "fixed-point", None, "3"),
        ("(1 + 1/2) * 3", "floating-point", None, "4.5"),
        ("(1 + 1/2) * 3", "fixed-point", 2, "4.50"),
        ("(0.1 + 0.2) * 3", "rational", None, "9/10"),
        ("0xDE0B6B3A7640000@18 / 4", "fixed-point", None, "0.250000000000000000"),
        ("2**10 / 8 - 1", "rational", None, "127"),
        ("100 // 7 + 100 % 7", "fixed-point", None, "16"),
    ]
    payloads = _analyze_in_one_session([(e, m, f) for e, m, f, _ in cases])

    # Under --verbose/--human-readable, show the tool's input and the actual trees.
    if _tracing(request):
        with capsys.disabled():
            for _case, (arguments, payload) in zip(cases, payloads, strict=True):
                print(f"\nINPUT  analyze({arguments})")
                print("OUTPUT tree:")
                print(payload["tree"])

    for (expr, _mode, _floor, root), (_arguments, payload) in zip(cases, payloads, strict=True):
        assert payload["error"] is None, f"{expr!r} errored: {payload['error']}"
        first_line = payload["tree"].splitlines()[0]
        # Each line is `<OPCODE/LITERAL> Value = <value> (<type>[scale], <verdict>)[ · details]`
        # (26): the rendered value sits between " = " and the " (" type envelope.
        rendered = first_line.split(" = ", 1)[1].split(" (", 1)[0]
        assert rendered == root, f"{expr!r} root mismatch: {first_line!r}"
        # Every non-leaf line carries its computed sub-result, so each has " = ".
        assert all(" = " in line for line in payload["tree"].splitlines())


def test_analyze_renders_bitwise_trees_over_the_wire(capsys, request):
    # The bitwise ops (24.3.2) through the real `analyze` path: the printed AST
    # pins the opcodes (BINARY_AND/OR/XOR, UNARY_NOT), the | < ^ < & precedence
    # and ~-as-tight-prefix shape, and each node's per-mode value / exactness / bits:
    #   - fixed-point shows the negative sign-extend (~5 -> -6) and hex mantissa;
    #   - fixed-point ^ across differing scales lands at the max() scale (1.5 ^ 3);
    #   - rational is componentwise on num/den ((3/2) ^ (5/4) = 6/6 = 1);
    #   - floating-point masks the raw IEEE-754 bit pattern (2.0 | 1.0 -> inf).
    cases = [
        (
            "~5 & 0xF0 | 3",
            "fixed-point",
            "BINARY_OR Value = 243 (fixed-point[0], exact) · hex 0xf3\n"
            "  BINARY_AND Value = 240 (fixed-point[0], exact) · hex 0xf0\n"
            "    UNARY_NOT Value = -6 (fixed-point[0], exact) · hex -0x06\n"
            '      LITERAL "5" Value = 5 (fixed-point[0], exact) · hex 0x05\n'
            '    LITERAL "0xF0" Value = 240 (fixed-point[0], exact) · hex 0xf0\n'
            '  LITERAL "3" Value = 3 (fixed-point[0], exact) · hex 0x03',
        ),
        (
            "1.5 ^ 3",
            "fixed-point",
            "BINARY_XOR Value = 1.7 (fixed-point[1], exact) · hex 0x11@1\n"
            '  LITERAL "1.5" Value = 1.5 (fixed-point[1], exact) · hex 0x0f@1\n'
            '  LITERAL "3" Value = 3 (fixed-point[0], exact) · hex 0x03',
        ),
        (
            "1.5 ^ 1.25",
            "rational",
            "BINARY_XOR Value = 1 (rational, exact)\n"
            '  LITERAL "1.5" Value = 3/2 (rational, exact) · ≈ 1.5\n'
            '  LITERAL "1.25" Value = 5/4 (rational, exact) · ≈ 1.25',
        ),
        (
            "2.0 | 1.0",
            "floating-point",
            "BINARY_OR Value = inf (floating-point, inexact) · hex 0x7ff0000000000000\n"
            '  LITERAL "2.0" Value = 2.0 (floating-point, inexact) · hex 0x4000000000000000\n'
            '  LITERAL "1.0" Value = 1.0 (floating-point, inexact) · hex 0x3ff0000000000000',
        ),
    ]
    payloads = _analyze_in_one_session([(e, m, None) for e, m, _ in cases])

    # Under --verbose/--human-readable, show the tool's input and the actual trees.
    if _tracing(request):
        with capsys.disabled():
            for (_expr, _mode, _tree), (arguments, payload) in zip(cases, payloads, strict=True):
                print(f"\nINPUT  analyze({arguments})")
                print("OUTPUT tree:")
                print(payload["tree"])

    for (expr, _mode, expected_tree), (_arguments, payload) in zip(cases, payloads, strict=True):
        assert payload["error"] is None, f"{expr!r} errored: {payload['error']}"
        assert payload["tree"] == expected_tree, f"{expr!r} tree mismatch:\n{payload['tree']}"


def test_analyze_rational_bitwise_zero_denominator_errors_over_the_wire():
    # Componentwise rational bitwise (24.3.2) can drive a denominator to zero:
    # 5 ^ 3 is (5 ^ 3) / (1 ^ 1) = 6 / 0. It must surface as a domain error with a
    # null tree, NOT a protocol failure — the price the all-types-bits decision
    # accepted for rational's faithful numerator/denominator componentwise.
    ((_arguments, payload),) = _analyze_in_one_session([("5 ^ 3", "rational", None)])
    assert payload["tree"] is None
    assert payload["error"]  # a plain diagnostic, no "error (line N):" prefix


def test_initialize_handshake_identifies_the_server():
    async def go():
        async with _client() as session:
            # A successful _client() already completed initialize(); re-issuing
            # a request proves the session is live after the handshake.
            return await session.list_tools()

    tools = asyncio.run(go())
    assert sorted(t.name for t in tools.tools) == [
        "analyze",
        "calculate",
        "help",
        "info",
        "solver",
    ]


def test_reference_is_exposed_as_resources_over_the_wire():
    # TODO 41.8: the help reference is also exposed as MCP resources — a static index
    # (abacus://reference) plus a template (abacus://reference/{section}) — so Glama's
    # resources probe finds something and a client can read the grammar without a tool
    # call. Asserted end-to-end: the registrations are advertised AND readable.
    async def go():
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resources = await session.list_resources()
                templates = await session.list_resource_templates()
                index = await session.read_resource(AnyUrl("abacus://reference"))
                types = await session.read_resource(AnyUrl("abacus://reference/types"))
                return resources, templates, index, types

    resources, templates, index, types = asyncio.run(go())
    assert "abacus://reference" in [str(r.uri) for r in resources.resources]
    assert "abacus://reference/{section}" in [t.uriTemplate for t in templates.resourceTemplates]
    # The index lists every section; a section read returns that section's text.
    index_text = index.contents[0].text
    for section in ("types", "language", "functions", "solver"):
        assert section in index_text
    assert "floating-point" in types.contents[0].text  # the 'types' section content


def test_initialize_carries_server_instructions_over_the_wire():
    # TODO 41.3: the server provides an MCP-level `instructions` string, so a client
    # gets a server overview (the mental model: type-faithful modes + precision
    # verdict) from the initialize handshake, not only from per-tool docstrings.
    async def go():
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                return await session.initialize()

    instructions = asyncio.run(go()).instructions
    assert instructions  # present and non-empty
    # The overview names the three modes and the precision-verdict contract.
    for mode in ("fixed-point", "floating-point", "rational"):
        assert mode in instructions
    assert "precision verdict" in instructions
    # ...and the five tools the server exposes.
    for tool in ("calculate", "analyze", "solver", "help", "info"):
        assert tool in instructions


def test_calculate_is_advertised_with_optional_mode_over_the_wire():
    async def go():
        async with _client() as session:
            return await session.list_tools()

    tools = {t.name: t for t in asyncio.run(go()).tools}
    schema = tools["calculate"].inputSchema
    assert schema["required"] == ["expression"]  # mode is optional...
    assert schema["properties"]["mode"]["default"] == "fixed-point"  # ...and defaults here


def test_every_tool_parameter_is_described_in_its_schema_over_the_wire():
    # TODO 41.1: every tool parameter must carry its own inputSchema `description`
    # (the per-parameter docs FastMCP builds from `Annotated[..., Field(...)]`), not
    # just the prose buried in the tool description — this is the schema coverage a
    # client (and Glama's grader) reads. Asserted end-to-end so a regression to bare
    # signature args, which silently drops the descriptions, fails here.
    async def go():
        async with _client() as session:
            return await session.list_tools()

    tools = {t.name: t for t in asyncio.run(go()).tools}
    # `info` is nullary; the other four expose parameters that must all be described.
    expected_param_counts = {"analyze": 3, "calculate": 4, "help": 1, "solver": 9}
    for name, count in expected_param_counts.items():
        properties = tools[name].inputSchema["properties"]
        assert len(properties) == count, f"{name}: param count changed to {len(properties)}"
        undocumented = [
            param for param, schema in properties.items() if not schema.get("description")
        ]
        assert not undocumented, f"{name} has parameters with no schema description: {undocumented}"
    # `info` takes no arguments, so it has nothing to document.
    assert tools["info"].inputSchema.get("properties", {}) == {}


def test_calculate_description_points_to_its_sibling_tools_over_the_wire():
    # TODO 41.5: calculate's description must steer the model to the right tool —
    # `analyze` to see WHERE an answer rounded, `solver` to find an unknown — so the
    # cross-references are no longer one-directional. Asserted on the wire.
    async def go():
        async with _client() as session:
            return await session.list_tools()

    description = {t.name: t for t in asyncio.run(go()).tools}["calculate"].description
    assert "analyze" in description and "solver" in description
    assert "Use `calculate`" in description  # the explicit when-to-use guidance


def test_solver_param_descriptions_encode_the_single_vs_multiple_coupling():
    # TODO 41.6: the schema shows variable/lower/upper and variables as independent
    # optionals (no oneOf), so the "give exactly one form" coupling must live in the
    # per-parameter descriptions. Asserted on the wire for both forms' anchors.
    async def go():
        async with _client() as session:
            return await session.list_tools()

    props = {t.name: t for t in asyncio.run(go()).tools}["solver"].inputSchema["properties"]
    variable_desc = props["variable"]["description"]
    variables_desc = props["variables"]["description"]
    # Each anchor names the OTHER form and states the exactly-one rule.
    assert "variables" in variable_desc and "EXACTLY ONE" in variable_desc
    assert "variable" in variables_desc and "EXACTLY ONE" in variables_desc
    # The bracket bounds tie themselves to the single-form trio.
    assert "variables" in props["lower"]["description"]
    assert "variables" in props["upper"]["description"]


def test_help_section_is_advertised_as_an_enum_over_the_wire():
    # TODO 41.2: help's `section` is a Literal, so its inputSchema must carry an
    # `enum` of the four valid sections — what lets a client pick a valid value
    # without guessing. Asserted end-to-end so the enum survives the wire.
    async def go():
        async with _client() as session:
            return await session.list_tools()

    tools = {t.name: t for t in asyncio.run(go()).tools}
    section = tools["help"].inputSchema["properties"]["section"]
    assert sorted(section["enum"]) == ["functions", "language", "solver", "types"]


def test_calculate_defaults_to_fixed_point_over_stdio():
    # 0.1 + 0.2 is the flagship discriminator: it is 0.3 in scaled fixed-point,
    # but the famous 0.30000000000000004 in floating_point. Omitting `mode` must pick
    # the money-safe default all the way through the protocol.
    result = _call("calculate", {"expression": "0.1 + 0.2"})
    assert result.isError is False
    assert _value(result) == "0.3"


def test_each_mode_evaluates_the_same_expression_its_own_way():
    # One session, one expression, three regimes — the product's whole pitch is
    # that the *type* governs the arithmetic, and that must hold over the wire.
    modes = ("fixed-point", "floating-point", "rational")
    texts = _calc_in_one_session([("0.1 + 0.2", m) for m in modes])
    assert dict(zip(modes, texts, strict=True)) == {
        "fixed-point": "0.3",
        "floating-point": "0.30000000000000004",
        "rational": "3/10",
    }


def test_operator_precedence_and_mixed_bases_travel_over_the_wire():
    # Each expression pins a piece of the grammar end-to-end through the
    # protocol; all are integer-valued, so the default fixed-point mode suffices.
    cases = {
        "2**3**2": "512",  # power is right-associative -> 2**(3**2)
        "-2**2": "-4",  # power binds tighter than unary minus -> -(2**2)
        "7 % 3 ** 2": "7",  # ...and tighter than % -> 7 % 9
        "((2 + 3) * (4 - 1))**2": "225",  # nested grouping -> 15**2
        "100 // 7 + 100 % 7": "16",  # floor-division and modulo -> 14 + 2
        "0xFF * 0b10 + 0o17": "525",  # hex, binary, octal literals in one expr
    }
    texts = _calc_in_one_session([(expr, None) for expr in cases])
    assert dict(zip(cases, texts, strict=True)) == cases


def test_bitwise_operators_travel_over_the_wire():
    # The bitwise rungs end-to-end (24.3.2): ^ is XOR (not power), the | < ^ < &
    # ordering below additive, and ~ as a tight unary prefix. Integer-valued, so
    # the default fixed-point mode suffices.
    cases = {
        "0xF0 | 0x0F": "255",  # OR combines the nibbles -> 0xFF
        "0b1100 & 0b1010": "8",  # AND -> 0b1000
        "5 ^ 3": "6",  # XOR -> 0b110 (NOT power: 5 ** 3 would be 125)
        "2 ^ 3": "1",  # ^ is XOR; contrast 2 ** 3 below
        "2 ** 3": "8",  # ** is still power
        "1 | 2 ^ 3 & 4 + 5": "3",  # 1 | (2 ^ (3 & (4 + 5))) = 1 | (2 ^ 1) = 1 | 3
        "~5 + 8": "2",  # ~ binds tighter than + : (~5) + 8 = -6 + 8
        "~0": "-1",  # ~x == -x-1 on the integer mantissa
    }
    texts = _calc_in_one_session([(expr, None) for expr in cases])
    assert dict(zip(cases, texts, strict=True)) == cases


def test_binary_and_octal_literals_evaluate_over_the_wire():
    # Exercise the 0b / 0o literal forms directly: standalone values, mixed-base
    # arithmetic, and cross-base equivalences (0xFF == 0o377, 0b100 == 0o4)
    # proving every prefix parses to the same integer. All integer-valued, so
    # the default fixed-point mode suffices.
    cases = {
        "0b1010": "10",  # binary -> 10
        "0o17": "15",  # octal -> 15
        "0b11111111": "255",  # a binary byte -> 255
        "0o777": "511",  # octal -> 511
        "0o10 * 0b10": "16",  # octal 8 * binary 2
        "(0b1010 + 0o12) * 0o2": "40",  # mixed bases with grouping -> (10 + 10) * 2
        "0o20 // 0b11": "5",  # floor-division of octal 16 by binary 3
        "0b1010 % 0o11": "1",  # binary 10 mod octal 9
        "0xFF - 0o377": "0",  # 255 in hex equals 255 in octal
        "0b100 - 0o4": "0",  # 4 in binary equals 4 in octal
    }
    texts = _calc_in_one_session([(expr, None) for expr in cases])
    assert dict(zip(cases, texts, strict=True)) == cases


def test_complex_expressions_diverge_by_mode():
    # Richer expressions where each regime's character shows: float rounding,
    # fixed-point truncation, and rational exactness, all over the wire.
    modes = ("fixed-point", "floating-point", "rational")
    expected = {
        "(0.1 + 0.2) * 3": {
            "fixed-point": "0.9",
            "floating-point": "0.9000000000000001",
            "rational": "9/10",
        },
        "1 / 7 * 7": {"fixed-point": "0", "floating-point": "1.0", "rational": "1"},
        "2**-3": {"fixed-point": "0", "floating-point": "0.125", "rational": "1/8"},
        "2.5e-4 * 1e4": {"fixed-point": "2.50000", "floating-point": "2.5", "rational": "5/2"},
    }
    calls = [(expr, m) for expr in expected for m in modes]
    texts = _calc_in_one_session(calls)
    got = {
        expr: dict(zip(modes, texts[i * 3 : i * 3 + 3], strict=True))
        for i, expr in enumerate(expected)
    }
    assert got == expected


def test_ethereum_wei_literal_evaluates_across_modes():
    # 0xDE0B6B3A7640000@18 is 1 ETH as wei: a hex mantissa scaled by 10^-18.
    # Quartering it stays exact in fixed-point and rational but rounds in float.
    modes = ("fixed-point", "floating-point", "rational")
    texts = _calc_in_one_session([("0xDE0B6B3A7640000@18 / 4", m) for m in modes])
    assert dict(zip(modes, texts, strict=True)) == {
        "fixed-point": "0.250000000000000000",
        "floating-point": "0.25",
        "rational": "1/4",
    }


def test_explicit_rational_mode_stays_exact():
    result = _call("calculate", {"expression": "1/3", "mode": "rational"})
    assert _value(result) == "1/3"


def test_precision_verdict_and_fields_travel_over_the_wire():
    # 25.1: every result must declare its precision so an inexact `/` cannot be
    # mistaken for exact. The annotated value AND the structured exact/precision
    # fields must survive the full stdio round-trip, for every mode.
    exact_fp = _payload(_call("calculate", {"expression": "2 + 3"}))
    assert exact_fp["value"] == "5 (exact)"
    assert exact_fp["exact"] is True and exact_fp["precision"] == 0

    inexact_fp = _payload(_call("calculate", {"expression": "10 / 3"}))
    assert inexact_fp["value"] == (
        "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
        "for more; e.g. =4 → 3.3333)"
    )
    assert inexact_fp["exact"] is False and inexact_fp["precision"] == 0
    # 27: mode, value_hex_dump and the offered_precision what-if survive the wire.
    assert inexact_fp["mode"] == "fixed-point"
    assert inexact_fp["value_hex_dump"] == "0x03"  # mantissa 3, scale 0
    assert inexact_fp["offered_precision"] == {
        "mode": "fixed-point",
        "min_fixed_point_precision": 4,
        "value": "3.3333 (inexact, rounded to 4 decimals)",
        "value_hex_dump": "0x8235@4",
        "exact": False,
    }

    flt = _payload(_call("calculate", {"expression": "0.1 + 0.2", "mode": "floating-point"}))
    assert flt["value"] == "0.30000000000000004 (inexact)"
    assert flt["exact"] is False and flt["precision"] is None
    # 27.1/27.5: float reads back as its canonical mode and dumps its raw IEEE bits.
    assert flt["mode"] == "floating-point"
    assert flt["value_hex_dump"] == "0x3fd3333333333334"

    rat = _payload(_call("calculate", {"expression": "1/3", "mode": "rational"}))
    assert rat["value"] == "1/3 (exact)"
    assert rat["exact"] is True and rat["precision"] is None
    # 27.5: rational has no single integer to dump.
    assert rat["mode"] == "rational" and rat["value_hex_dump"] is None


def test_inexact_handling_abort_on_fixed_point_travels_over_the_wire():
    # 35.2.2 end-to-end over real stdio: with inexact_handling=abort-on-inexact a
    # rounded fixed-point division is REJECTED — the call still succeeds at the
    # protocol level (error as ordinary text), and the headline names the line and
    # lays the operation out in VALUES at the active precision (35.3.1).
    aborted = _payload(
        _call(
            "calculate",
            {"expression": "1.00 / 3.00", "inexact_handling": "abort-on-inexact"},
        )
    )
    assert aborted["value"] is None and aborted["exact"] is None
    error = aborted["error"]
    assert error.startswith("Inexact calculation in line 1: 1.00 / 3.00 = 0.33 is not exact.")

    # An EXACT fixed-point calculation passes through unchanged under the same policy.
    exact = _payload(_call("calculate", {"expression": "1 + 2", "inexact_handling": "abort"}))
    assert exact["value"] == "3 (exact)" and exact["exact"] is True and exact["error"] is None

    # An unknown policy name lists the valid choices rather than failing the protocol.
    unknown = _payload(_call("calculate", {"expression": "1", "inexact_handling": "perhaps"}))
    assert unknown["value"] is None
    assert "Unknown inexact_handling" in unknown["error"]
    assert "continue-and-report" in unknown["error"] and "abort-on-inexact" in unknown["error"]


def test_floating_point_aliases_resolve_to_the_canonical_mode():
    # The AI may spell the floating-point mode "float64" or "double" (23.6);
    # all three resolve to the same IEEE-754 double arithmetic, where the
    # flagship 0.1 + 0.2 famously rounds — while fixed-point holds it exactly,
    # the contrast the whole product exists to show.
    expr = "0.1 + 0.2"
    expected = {
        "floating-point": "0.30000000000000004",
        "float64": "0.30000000000000004",
        "double": "0.30000000000000004",
        "fixed-point": "0.3",
    }
    texts = _calc_in_one_session([(expr, name) for name in expected])
    assert dict(zip(expected, texts, strict=True)) == expected


def test_malformed_expression_returns_an_error_not_a_protocol_failure():
    result = _call("calculate", {"expression": "1 +"})
    # The tool reports user errors as ordinary text (a plain, self-contained
    # message), so the call itself succeeds at the protocol level.
    assert result.isError is False
    assert _payload(result)["error"] == "expected a number, name, or '(', got end of input"


def test_unknown_mode_lists_the_valid_modes():
    result = _call("calculate", {"expression": "1", "mode": "int128"})
    error = _payload(result)["error"]
    assert "Unknown mode" in error
    assert "fixed-point" in error and "floating-point" in error and "rational" in error


# --- solver over the wire (31.7 / 31.9) --------------------------------------


def _solve(arguments):
    """Invoke the `solver` tool against a fresh server; return its structured dict."""
    return _payload(_call("solver", arguments))


def _solve_in_one_session(cases):
    """Invoke `solver` for each argument dict against ONE server; return its payloads.

    Keeps a multi-call observability test on a single subprocess; order preserved.
    """

    async def go():
        async with _client() as session:
            payloads = []
            for arguments in cases:
                result = await session.call_tool("solver", arguments)
                payloads.append(_payload(result))
            return payloads

    return asyncio.run(go())


def test_solver_replies_are_observable_over_the_wire(capsys, request):
    # A spread of solver calls over real stdio, PRINTED under --verbose/--human-
    # readable (./run-tests.sh) to show each request and its full reply. The five
    # well-posed cases find a root or an extremum; the last is an honest no-solution.
    # Each case is (arguments, human note).
    cases = [
        (
            {"expression": "x**2 - 2", "variable": "x", "lower": 0, "upper": 2, "mode": "double"},
            "find-root -> sqrt(2), value ~ 0",
        ),
        (
            {
                "expression": "(x - 3)**2",
                "variable": "x",
                "lower": 0,
                "upper": 5,
                "objective": "find-minimum",
                "mode": "double",
            },
            "find-minimum of a unimodal expr -> x = 3, value 0",
        ),
        (
            {
                "expression": "5 - (x - 1)**2",
                "variable": "x",
                "lower": -2,
                "upper": 4,
                "objective": "max",
                "mode": "double",
            },
            "find-maximum (alias 'max') -> x = 1, value 5",
        ),
        (
            {
                "expression": "r = 0.05\np = 1000\np * (1 + r)**n - 2000",
                "variable": "n",
                "lower": 0,
                "upper": 100,
                "mode": "double",
            },
            "constants from assignment lines -> n ~ 14.21",
        ),
        (
            {
                "expression": "2*x - 3",
                "variable": "x",
                "lower": 0,
                "upper": 3,
                "mode": "fixed-point",
                "min_fixed_point_precision": 1,
            },
            "fixed-point exact root -> 1.5 (value exactly 0)",
        ),
        (
            {"expression": "x**2 + 1", "variable": "x", "lower": 0, "upper": 2, "mode": "double"},
            "no solution -> the closest |expr| is reported",
        ),
    ]
    payloads = _solve_in_one_session([arguments for arguments, _note in cases])

    if _tracing(request):
        with capsys.disabled():
            for (arguments, note), payload in zip(cases, payloads, strict=True):
                print(f"\nINPUT  solver({arguments})  # {note}")
                print("OUTPUT:")
                print(json.dumps(payload, indent=2))

    for (_arguments, note), payload in zip(cases[:-1], payloads[:-1], strict=True):
        assert payload["error"] is None, f"{note}: {payload['error']}"
    assert payloads[-1]["error"] is not None and "No solution" in payloads[-1]["error"]


def test_solver_finds_a_root_over_the_wire():
    # x**2 - 2 == 0 over [0, 2] in floating-point -> sqrt(2); the reply carries the
    # solution, the (near-zero) expression value there, and the resolved objective.
    payload = _solve(
        {"expression": "x**2 - 2", "variable": "x", "lower": 0, "upper": 2, "mode": "double"}
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "golden-section-search"
    assert payload["solution"].startswith("1.4142")
    assert "(approximate)" in payload["solution"]
    assert abs(float(payload["value"].split(" (")[0])) < 1e-6
    assert payload["mode"] == "floating-point"
    assert payload["iterations"] > 0


def test_solver_finds_a_maximum_over_the_wire():
    # 5 - (x - 1)**2 peaks at x == 1 with value 5; find-maximum drives there, and the
    # alias "max" resolves to the canonical find-maximum.
    payload = _solve(
        {
            "expression": "5 - (x - 1)**2",
            "variable": "x",
            "lower": -2,
            "upper": 4,
            "objective": "max",
            "mode": "double",
        }
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-maximum"
    assert float(payload["solution"].split(" (")[0]) == pytest.approx(1.0, abs=1e-4)
    assert float(payload["value"].split(" (")[0]) == pytest.approx(5.0, abs=1e-6)


def test_solver_reads_constants_from_assignments_over_the_wire():
    # The program sets r, p; n is the unknown. Newlines travel intact across the
    # wire, so the multi-line program solves p*(1+r)**n == 2000 for n.
    payload = _solve(
        {
            "expression": "r = 0.05\np = 1000\np * (1 + r)**n - 2000",
            "variable": "n",
            "lower": 0,
            "upper": 100,
            "mode": "double",
        }
    )
    assert payload["error"] is None
    assert float(payload["solution"].split(" (")[0]) == pytest.approx(14.2067, abs=1e-3)


def test_solver_no_solution_reports_the_closest_residual():
    # x**2 + 1 never reaches zero on [0, 2]; the error names it and the closest |expr|.
    payload = _solve(
        {"expression": "x**2 + 1", "variable": "x", "lower": 0, "upper": 2, "mode": "double"}
    )
    assert payload["solution"] is None and payload["value"] is None
    assert "No solution" in payload["error"] and "closest" in payload["error"]


def test_solver_rejects_a_variable_absent_from_the_expression():
    payload = _solve({"expression": "y + 1", "variable": "x", "lower": 0, "upper": 2})
    assert payload["solution"] is None
    assert "does not occur" in payload["error"]


def test_solver_rejects_an_empty_bracket():
    payload = _solve({"expression": "x - 1", "variable": "x", "lower": 2, "upper": 2})
    assert payload["solution"] is None
    assert "bracket is empty" in payload["error"]


def test_solver_rejects_an_unknown_mode():
    payload = _solve(
        {"expression": "x - 1", "variable": "x", "lower": 0, "upper": 2, "mode": "int128"}
    )
    assert payload["solution"] is None
    assert "Unknown mode" in payload["error"]


def test_solver_rejects_min_fixed_point_precision_outside_fixed_point():
    payload = _solve(
        {
            "expression": "x - 1",
            "variable": "x",
            "lower": 0,
            "upper": 2,
            "mode": "double",
            "min_fixed_point_precision": 4,
        }
    )
    assert payload["solution"] is None
    assert "only valid in fixed-point and complex modes" in payload["error"]


def test_solver_rejects_an_unknown_objective():
    # Bare "optimise" no longer names an objective (the direction lives in find-min/max).
    payload = _solve(
        {"expression": "x - 1", "variable": "x", "lower": 0, "upper": 2, "objective": "optimise"}
    )
    assert payload["solution"] is None
    assert "Unknown objective" in payload["error"]


def test_solver_finds_a_multivariate_minimum_over_the_wire():
    # The Nelder-Mead engine drives TWO unknowns jointly: (x-3)**2 + (y+1)**2 has its
    # minimum at (3, -1), value 0. The `variables` form and the `solutions` list both
    # travel intact across the wire.
    payload = _solve(
        {
            "expression": "(x - 3)**2 + (y + 1)**2",
            "variables": {"x": [0, 5], "y": [-4, 2]},
            "objective": "find-minimum",
            "algorithm": "nelder-mead",
            "mode": "double",
        }
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-minimum"
    assert payload["algorithm"] == "nelder-mead"
    # Multivariate: the scalar solution is null; every unknown is in `solutions`.
    assert payload["solution"] is None
    found = {
        entry["variable"]: float(entry["solution"].split(" (")[0]) for entry in payload["solutions"]
    }
    assert found["x"] == pytest.approx(3.0, abs=1e-3)
    assert found["y"] == pytest.approx(-1.0, abs=1e-3)
    assert float(payload["value"].split(" (")[0]) == pytest.approx(0.0, abs=1e-6)


def test_solver_rejects_multiple_unknowns_for_golden_section():
    # The `variables` form needs Nelder-Mead; golden-section (the default) is single-
    # variable, so the request is refused with a pointer to the right algorithm.
    payload = _solve({"expression": "x + y", "variables": {"x": [0, 1], "y": [0, 1]}})
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]
