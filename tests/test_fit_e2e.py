# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end functional tests for the `curve_fit` tool (TODO 44).

The companion to test_fit.py's unit-level coverage: where that drives the module's
engine directly, this exercises the WHOLE seam — a request dict goes in as the tool's
arguments and the structured reply comes back, the same path a real client takes
(dispatch -> validate -> fit -> render). The in-process FastMCP `mcp.call_tool` path
is used (no subprocess); test_e2e.py covers the tools over real stdio.
"""

import asyncio
import json

from mcp_abacus.server import mcp


def _fit(x, y, *, mode=None, floor=None):
    """Invoke the `curve_fit` tool in-process; return its structured reply dict (44.1).

    Optional arguments are omitted from the request when None, so the tool's own
    defaults (fixed-point mode, floor 9) are exercised.
    """
    arguments = {"x": x, "y": y}
    if mode is not None:
        arguments["mode"] = mode
    if floor is not None:
        arguments["min_fixed_point_precision"] = floor
    result = asyncio.run(mcp.call_tool("curve_fit", arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def test_linear_fit_reply_shape():
    payload = _fit([1, 2, 3, 4], [3, 5, 7, 9], mode="rational")
    assert payload["error"] is None
    assert payload["mode"] == "rational"
    assert len(payload["fits"]) == 1
    fit = payload["fits"][0]
    assert fit["form"] == "linear"
    assert fit["equation"] == "2*x + 1"
    assert [p["name"] for p in fit["parameters"]] == ["a", "b"]
    assert fit["parameters"][0]["value"] == "2 (exact)"
    assert fit["parameters"][1]["value"] == "1 (exact)"
    assert fit["fit_error"] == "0 (exact)"
    assert fit["exact"] is True


def test_default_mode_is_fixed_point_with_a_sub_unit_floor():
    # Omitting mode/floor: fixed-point at the default floor 9, so the parameters keep
    # nine fractional digits rather than rounding to whole numbers.
    payload = _fit([1, 1.5, 2.0], [2, 5.8, 8.9])
    assert payload["mode"] == "fixed-point"
    fit = payload["fits"][0]
    # slope 6.9, intercept -4.78333... at scale 9.
    assert fit["parameters"][0]["value"].startswith("6.900000000")
    assert fit["parameters"][1]["value"].startswith("-4.783333333")
    assert fit["equation"].startswith("6.900000000*x - 4.783333333")


def test_length_mismatch_is_an_error():
    payload = _fit([1, 2, 3], [1, 2])
    assert payload["fits"] is None and payload["mode"] is None
    assert "same length" in payload["error"]


def test_too_few_points_is_an_error():
    payload = _fit([1], [1])
    assert payload["fits"] is None
    assert "at least two" in payload["error"]


def test_vertical_line_is_an_error():
    payload = _fit([2, 2, 2], [1, 2, 3])
    assert payload["fits"] is None
    assert "every x value is equal" in payload["error"].lower()


def test_complex_mode_is_rejected():
    payload = _fit([1, 2], [1, 2], mode="complex")
    assert payload["fits"] is None
    assert "complex mode is not supported" in payload["error"]


def test_unknown_mode_lists_the_valid_modes():
    payload = _fit([1, 2], [1, 2], mode="bogus")
    assert payload["fits"] is None
    assert "Unknown mode" in payload["error"]


def test_min_fixed_point_precision_rejected_outside_fixed_point():
    payload = _fit([1, 2], [1, 2], mode="floating-point", floor=4)
    assert payload["fits"] is None
    assert "min_fixed_point_precision" in payload["error"]
