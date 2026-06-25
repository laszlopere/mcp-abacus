# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Prove the `info` tool is wired into the FastMCP app, not merely importable."""

import asyncio
import json

from mcp_abacus.expr import reference
from mcp_abacus.expr.value import Mode
from mcp_abacus.server import _resolve_mode_and_precision, mcp, reference_section


def _content_blocks(call_tool_result):
    # FastMCP may return either a sequence of content blocks or a
    # (blocks, structured_output) tuple depending on SDK version.
    if isinstance(call_tool_result, tuple):
        return call_tool_result[0]
    return call_tool_result


def _calc(call_tool_result):
    # calculate returns a structured dict (25.1); FastMCP renders it as JSON text.
    return json.loads(_content_blocks(call_tool_result)[0].text)


def test_exposed_tools_are_info_help_calculate_analyze_and_solver():
    tools = asyncio.run(mcp.list_tools())
    assert sorted(t.name for t in tools) == ["analyze", "calculate", "help", "info", "solver"]


def test_tools_have_description_and_input_schema():
    tools = asyncio.run(mcp.list_tools())
    for tool in tools:
        assert tool.description
        assert tool.inputSchema
        assert tool.inputSchema.get("type") == "object"


def test_help_tool_takes_a_required_section_argument():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["help"].inputSchema
    assert "section" in schema["properties"]
    assert schema["required"] == ["section"]


def test_invoking_help_through_the_app_returns_section_text():
    result = asyncio.run(mcp.call_tool("help", {"section": "types"}))
    blocks = _content_blocks(result)
    assert "floating-point" in blocks[0].text


def test_reference_resources_are_registered_on_the_app():
    # TODO 41.8: the help reference is also exposed as resources, so a non-tool client
    # (and Glama's introspection) finds something under resources/templates.
    resources = {str(r.uri) for r in asyncio.run(mcp.list_resources())}
    templates = {t.uriTemplate for t in asyncio.run(mcp.list_resource_templates())}
    assert "abacus://reference" in resources
    assert "abacus://reference/{section}" in templates


def test_reference_section_resource_mirrors_the_help_tool():
    # The resource and the `help` tool render the SAME section text — one source.
    for section in reference.sections():
        assert reference_section(section) == reference.render(section)


def test_calculate_tool_has_optional_mode_defaulting_to_fixed_point():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["calculate"].inputSchema
    assert "expression" in schema["properties"]
    assert schema["required"] == ["expression"]  # mode is optional
    assert schema["properties"]["mode"]["default"] == "fixed-point"


def test_calculate_defaults_to_fixed_point():
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1 + 2 * 10**3"})))
    assert result["value"] == "2001 (exact)"
    assert result["exact"] is True
    assert result["precision"] == 0  # whole-number fixed-point: a known 0-decimal scale
    assert result["error"] is None


def test_calculate_honours_the_mode_argument():
    result = _calc(
        asyncio.run(mcp.call_tool("calculate", {"expression": "1/3", "mode": "rational"}))
    )
    assert result["value"] == "1/3 (exact)"
    assert result["exact"] is True
    assert result["precision"] is None  # rational has no decimal scale


def test_calculate_surfaces_inexact_fixed_point_with_its_precision():
    # The 25.1 trap: a `/` that rounds must NOT read as exact. The value string
    # carries the verdict and the scale; 25.3.2 steers toward more fixed-point
    # precision and 25.3.3 works the steer out inline. The fields still report the
    # scale it rounded at.
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "10 / 3"})))
    assert result["value"] == (
        "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
        "for more; e.g. =4 → 3.3333)"
    )
    assert result["exact"] is False
    assert result["precision"] == 0


def test_calculate_unknown_mode_lists_valid_modes():
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1", "mode": "int128"})))
    assert result["value"] is None and result["exact"] is None and result["precision"] is None
    error = result["error"]
    assert "Unknown mode" in error
    assert "fixed-point" in error and "floating-point" in error and "rational" in error


def test_calculate_min_fixed_point_precision_is_optional_in_the_schema():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["calculate"].inputSchema
    assert "min_fixed_point_precision" in schema["properties"]
    assert schema["required"] == ["expression"]  # the floor is optional


def test_shared_front_end_resolves_a_mode_and_aliases():
    # The shared mode/precision helper calculate, analyze, and solver all reuse.
    assert _resolve_mode_and_precision("rational", None) == (Mode.RATIONAL, None)
    assert _resolve_mode_and_precision("double", None) == (Mode.FLOATING_POINT, None)


def test_shared_front_end_reports_unknown_mode_and_bad_precision():
    mode, error = _resolve_mode_and_precision("int128", None)
    assert mode is None and "Unknown mode" in error
    mode, error = _resolve_mode_and_precision("rational", 4)
    assert mode is None and "only valid in fixed-point and complex modes" in error
    mode, error = _resolve_mode_and_precision("fixed-point", -1)
    assert mode is None and "non-negative integer" in error


def test_unknown_mode_suggests_the_nearest_spelling():
    # 43.5: a near-miss mode gets a "did you mean" pointing at the closest valid spelling
    # (canonical name or alias), alongside the full valid list.
    _, error = _resolve_mode_and_precision("fixedpoint", None)
    assert "Did you mean 'fixed-point'?" in error
    _, error = _resolve_mode_and_precision("doubl", None)  # alias near-miss ("double")
    assert "Did you mean 'double'?" in error


def test_unknown_mode_far_miss_gets_no_suggestion():
    # An unrelated word clears no candidate above the cutoff, so only the valid list is
    # offered — never a misleading suggestion.
    _, error = _resolve_mode_and_precision("zzz", None)
    assert "Did you mean" not in error
    assert "Valid modes" in error


def test_calculate_inexact_handling_is_optional_defaulting_to_continue():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["calculate"].inputSchema
    assert schema["properties"]["inexact_handling"]["default"] == "continue-and-report"
    assert schema["required"] == ["expression"]  # the policy is optional


def test_calculate_abort_on_inexact_fails_with_a_diagnostic():
    # 35.2.2 / 35.3.1: the caller asked for exact-only, so a rounded division is
    # rejected and the headline names the line and the operation in VALUES.
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "calculate",
                {"expression": "1.00 / 3.00", "inexact_handling": "abort-on-inexact"},
            )
        )
    )
    assert result["value"] is None and result["exact"] is None
    error = result["error"]
    assert "line 1" in error
    assert "1.00 / 3.00 = 0.33 is not exact" in error


def test_calculate_abort_on_inexact_passes_an_exact_result_through():
    # An exact calculation under the abort policy returns normally.
    result = _calc(
        asyncio.run(
            mcp.call_tool("calculate", {"expression": "1 + 2", "inexact_handling": "abort"})
        )
    )
    assert result["value"] == "3 (exact)" and result["exact"] is True and result["error"] is None


def test_calculate_unknown_inexact_handling_lists_valid_values():
    result = _calc(
        asyncio.run(mcp.call_tool("calculate", {"expression": "1", "inexact_handling": "maybe"}))
    )
    assert result["value"] is None
    error = result["error"]
    assert "Unknown inexact_handling" in error
    assert "continue-and-report" in error and "abort-on-inexact" in error


def test_solver_tool_signature_exposes_both_unknown_forms():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["solver"].inputSchema
    properties = schema["properties"]
    for name in (
        "expression",
        "variable",
        "lower",
        "upper",
        "variables",
        "objective",
        "algorithm",
    ):
        assert name in properties
    # Only `expression` is required: the unknown is given EITHER as variable+lower+upper
    # OR as the variables map, a choice validated at call time, so neither form's fields
    # are schema-required (33.14). objective/algorithm/mode stay optional.
    assert sorted(schema["required"]) == ["expression"]
    assert properties["mode"]["default"] == "fixed-point"


def test_calculate_floor_steers_inexact_division_to_more_fixed_point_decimals():
    # The 25.2 fix: instead of the bare whole-number rounding, the floor keeps the
    # result in fixed-point at a higher precision (the TODO's own example).
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "calculate",
                {"expression": "928347569 / 2345", "min_fixed_point_precision": 4},
            )
        )
    )
    assert result["value"] == "395883.8247 (inexact, rounded to 4 decimals)"
    assert result["exact"] is False
    assert result["precision"] == 4
    assert result["error"] is None


def test_calculate_inexact_steer_only_when_no_floor_was_given():
    # The "pass min_fixed_point_precision for more" steer is for callers who have
    # not engaged the knob; once a floor is passed they already know it, so the
    # verdict drops the steer (and the worked-example offer).
    bare = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "10 / 3"})))
    assert "pass min_fixed_point_precision for more" in bare["value"]
    floored = _calc(
        asyncio.run(
            mcp.call_tool("calculate", {"expression": "10 / 3", "min_fixed_point_precision": 6})
        )
    )
    assert floored["value"] == "3.333333 (inexact, rounded to 6 decimals)"
    assert "min_fixed_point_precision" not in floored["value"]


def test_calculate_floor_resolves_through_a_fixed_point_alias():
    # 'decimal' is a fixed-point alias, so the floor is accepted there too.
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "calculate",
                {"expression": "10 / 3", "mode": "decimal", "min_fixed_point_precision": 3},
            )
        )
    )
    assert result["value"] == "3.333 (inexact, rounded to 3 decimals)"
    assert result["precision"] == 3


def test_calculate_floor_rejected_outside_fixed_point_mode():
    # Accepted ONLY in fixed-point mode; the error names the canonical mode.
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "calculate",
                {"expression": "1 / 3", "mode": "float", "min_fixed_point_precision": 4},
            )
        )
    )
    assert result["value"] is None and result["exact"] is None and result["precision"] is None
    assert "min_fixed_point_precision" in result["error"]
    assert "fixed-point" in result["error"] and "floating-point" in result["error"]


def test_calculate_floor_must_be_non_negative():
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "calculate",
                {"expression": "1 / 3", "min_fixed_point_precision": -1},
            )
        )
    )
    assert result["value"] is None
    assert "non-negative" in result["error"]


def test_calculate_floor_of_zero_matches_the_unfloored_result():
    # Floor 0 computes the SAME number as no floor — same value/exact/precision.
    # They differ only in the 27.2 offered_precision: an explicit floor (even 0)
    # means the caller already engaged the knob, so no what-if nudge; the bare call
    # gets one.
    floored = _calc(
        asyncio.run(
            mcp.call_tool("calculate", {"expression": "10 / 3", "min_fixed_point_precision": 0})
        )
    )
    bare = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "10 / 3"})))
    assert floored["exact"] == bare["exact"] and floored["precision"] == bare["precision"]
    # An explicit floor means the knob is engaged, so the verdict drops the steer.
    assert floored["value"] == "3 (inexact, rounded to 0 decimals)"
    assert floored["offered_precision"] is None
    # 27.3-27.5: the offered value carries its own mode, its OWN annotated verdict
    # (10/3 is still inexact at scale 4), and its hex dump (mantissa 33333 = 0x8235).
    # The offered preview is itself a floor, so its verdict also drops the steer.
    assert bare["offered_precision"] == {
        "mode": "fixed-point",
        "min_fixed_point_precision": 4,
        "value": "3.3333 (inexact, rounded to 4 decimals)",
        "value_hex_dump": "0x8235@4",
        "exact": False,
    }


def test_calculate_offers_a_higher_precision_for_inexact_fixed_point():
    # 25.3.3/27: the TODO's own example. An inexact fixed-point result with no floor
    # given offers the SAME expression at result scale + 4, both as a nested
    # offered_precision field and worked into the value string — so the hidden
    # digits are visible. The offered value is annotated (27.4) and carries its hex
    # dump (27.5): mantissa 3958838247 at scale 4 = 0xebf713e7@4.
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "928347569 / 2345"})))
    assert result["offered_precision"] == {
        "mode": "fixed-point",
        "min_fixed_point_precision": 4,
        "value": "395883.8247 (inexact, rounded to 4 decimals)",
        "value_hex_dump": "0xebf713e7@4",
        "exact": False,
    }
    # 27.6: the top-level value string KEEPS the inline worked example (bare digits).
    assert result["value"].endswith("; e.g. =4 → 395883.8247)")


def test_calculate_no_offer_when_the_floor_was_given():
    # The caller already engaged the knob — naming it back is no help, so no offer.
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "calculate",
                {"expression": "928347569 / 2345", "min_fixed_point_precision": 4},
            )
        )
    )
    assert result["offered_precision"] is None
    assert "e.g." not in result["value"]


def test_calculate_no_offer_for_exact_or_float_results():
    # Nothing to reveal when the result is exact, and float is the wrong direction.
    exact = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "2 + 3"})))
    assert exact["offered_precision"] is None
    flt = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "10 / 3", "mode": "float"})))
    assert flt["offered_precision"] is None


def test_calculate_no_offer_when_the_extra_digits_are_all_zero():
    # 25.3.3 gate: 1/10000000 rounds to 0 at scale 0 (inexact) but is still 0.0000
    # at scale 4 — the revealed digits are all zero, so an offer would only restate
    # the result with trailing zeros. Suppress it.
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1 / 10000000"})))
    assert result["exact"] is False  # genuinely rounded
    assert result["offered_precision"] is None
    assert "e.g." not in result["value"]


def test_calculate_returns_the_resolved_mode_even_through_an_alias():
    # 27.1: `mode` is the RESOLVED canonical Mode.value, so the reply stands on its
    # own — a request alias like "double" reads back as "floating-point".
    aliased = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1", "mode": "double"})))
    assert aliased["mode"] == "floating-point"
    # On success it equals the (canonical) request; on error it is null (27.1).
    fixed = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1"})))
    assert fixed["mode"] == "fixed-point"
    bad = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1", "mode": "int128"})))
    assert bad["mode"] is None


def test_calculate_returns_value_hex_dump_per_mode():
    # 27.5: the value's bits in hex beside the value string — fixed-point as M@D
    # (mantissa whole-byte hex, @scale dropped at scale 0), float as the raw 64-bit
    # IEEE-754 pattern, NULL in rational (no single integer to dump) and on error.
    fixed = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "100"})))
    assert fixed["value_hex_dump"] == "0x64"  # mantissa 100, scale 0
    scaled = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1.5"})))
    assert scaled["value_hex_dump"] == "0x0f@1"  # mantissa 15 at scale 1
    flt = _calc(
        asyncio.run(mcp.call_tool("calculate", {"expression": "0.1 + 0.2", "mode": "double"}))
    )
    assert flt["value_hex_dump"] == "0x3fd3333333333334"  # raw IEEE-754 bits
    rat = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1/3", "mode": "rational"})))
    assert rat["value_hex_dump"] is None
    err = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1 +"})))
    assert err["value_hex_dump"] is None


def test_calculate_inexact_float_is_not_steered_to_fixed_point_precision():
    # 25.3.2: only inexact FIXED-POINT steers — the arg is invalid in float mode,
    # and float is the wrong direction, so its bare "(inexact)" names nothing.
    result = _calc(
        asyncio.run(mcp.call_tool("calculate", {"expression": "10 / 3", "mode": "floating-point"}))
    )
    assert "min_fixed_point_precision" not in result["value"]
    assert result["value"].endswith("(inexact)")


def test_calculate_exact_fixed_point_is_not_steered():
    # An exact result has nothing to gain from more precision — no steer.
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "2 + 3"})))
    assert result["value"] == "5 (exact)"
    assert "min_fixed_point_precision" not in result["value"]


def test_calculate_reports_errors_as_a_plain_message():
    # The diagnostic stands on its own — no machine-style "error (line N):" prefix.
    result = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1 +"})))
    assert result["error"] == "expected a number, name, '(', or '[', got end of input"
    assert result["value"] is None and result["exact"] is None and result["precision"] is None


def test_tool_cross_references_are_bidirectional():
    # TODO 41.13: each of the three evaluation tools must steer to BOTH siblings, so an
    # LLM that lands on any one is pointed at the other two — not a one-way calculate ->
    # {analyze, solver} fan-out that strands a reader on analyze or solver.
    tools = {t.name: t.description for t in asyncio.run(mcp.list_tools())}
    siblings = {
        "calculate": ("analyze", "solver"),
        "analyze": ("calculate", "solver"),
        "solver": ("calculate", "analyze"),
    }
    for tool, refs in siblings.items():
        for ref in refs:
            assert f"`{ref}`" in tools[tool], f"{tool} never points to {ref}"


def test_analyze_tool_has_optional_mode_and_floor_defaulting_to_fixed_point():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["analyze"].inputSchema
    assert schema["required"] == ["expression"]  # mode and floor are optional
    assert schema["properties"]["mode"]["default"] == "fixed-point"
    assert "min_fixed_point_precision" in schema["properties"]


def test_analyze_returns_the_evaluated_tree_revealing_where_rounding_happened():
    # The point of analyze: (1 + 1/2) * 3 is 3, not 4.5, in fixed-point — and the
    # tree shows WHY, at the `1/2 = 0` leaf that rounded the half away at scale 0.
    # That leaf names HOW inexact too (34.5.2): rounding -1/2, the discarded half; the
    # nodes above it inherit the inexactness but introduce no rounding of their own.
    result = _calc(asyncio.run(mcp.call_tool("analyze", {"expression": "(1 + 1/2) * 3"})))
    assert result["error"] is None
    assert result["tree"] == (
        "BINARY_MUL Value = 3 (fixed-point[0], inexact) · hex 0x03\n"
        "  BINARY_ADD Value = 1 (fixed-point[0], inexact) · hex 0x01\n"
        '    LITERAL "1" Value = 1 (fixed-point[0], exact) · hex 0x01\n'
        "    BINARY_DIV Value = 0 (fixed-point[0], inexact) · hex 0x00 · rounding -1/2 ≈ -0.5\n"
        '      LITERAL "1" Value = 1 (fixed-point[0], exact) · hex 0x01\n'
        '      LITERAL "2" Value = 2 (fixed-point[0], exact) · hex 0x02\n'
        '  LITERAL "3" Value = 3 (fixed-point[0], exact) · hex 0x03'
    )


def test_analyze_honours_the_mode_argument():
    # In floating-point the same tree keeps the half — every node shows .0 forms.
    result = _calc(
        asyncio.run(
            mcp.call_tool("analyze", {"expression": "(1 + 1/2) * 3", "mode": "floating-point"})
        )
    )
    assert result["tree"].splitlines()[0] == (
        "BINARY_MUL Value = 4.5 (floating-point, inexact) · hex 0x4012000000000000"
    )
    assert (
        "BINARY_DIV Value = 0.5 (floating-point, inexact) · hex 0x3fe0000000000000"
        in (result["tree"])
    )


def test_analyze_threads_the_fixed_point_floor_into_the_tree():
    # The floor raises the scale of every leaf, so the rounding-away no longer happens.
    result = _calc(
        asyncio.run(
            mcp.call_tool(
                "analyze",
                {"expression": "(1 + 1/2) * 3", "min_fixed_point_precision": 2},
            )
        )
    )
    assert result["tree"].splitlines()[0] == (
        "BINARY_MUL Value = 4.50 (fixed-point[2], exact) · hex 0x01c2@2"
    )
    assert "BINARY_DIV Value = 0.50 (fixed-point[2], exact) · hex 0x32@2" in result["tree"]


def test_analyze_unknown_mode_returns_null_tree_with_the_mode_list():
    result = _calc(asyncio.run(mcp.call_tool("analyze", {"expression": "1", "mode": "int128"})))
    assert result["tree"] is None
    assert "Unknown mode" in result["error"]
    assert "fixed-point" in result["error"] and "rational" in result["error"]


def test_analyze_reports_malformed_expression_as_a_plain_message_with_null_tree():
    result = _calc(asyncio.run(mcp.call_tool("analyze", {"expression": "1 +"})))
    assert result["tree"] is None
    assert result["error"] == "expected a number, name, '(', or '[', got end of input"


def test_invoking_through_the_app_returns_the_info_payload():
    result = asyncio.run(mcp.call_tool("info", {}))
    blocks = _content_blocks(result)
    payload = json.loads(blocks[0].text)
    assert payload["status"] == "available"
    assert payload["name"] == "mcp-abacus"
    assert payload["version"]
    assert payload["python"]
    assert payload["mcp_sdk"]
    assert payload["toolsets"] == []


def test_calculate_reply_has_values_array():
    # The success reply carries a `values` list — one object per answered line — with a
    # fixed key set; the error reply nulls it like every other field (shape never varies).
    ok = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1 + 1\n2 + 2"})))
    assert isinstance(ok["values"], list) and len(ok["values"]) == 2
    for entry in ok["values"]:
        assert set(entry) == {
            "source",
            "value",
            "value_hex_dump",
            "exact",
            "precision",
            "offered_precision",
        }
    bad = _calc(asyncio.run(mcp.call_tool("calculate", {"expression": "1", "mode": "int128"})))
    assert bad["error"] is not None
    assert bad["values"] is None


def test_analyze_reply_unchanged_by_values():
    # analyze is untouched: it returns only `tree`/`error`, never a `values` array.
    result = _calc(asyncio.run(mcp.call_tool("analyze", {"expression": "1 + 2"})))
    assert set(result) == {"tree", "error"}
