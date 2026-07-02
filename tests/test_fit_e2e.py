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
import math

import pytest

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


def _by_form(payload):
    """Index the reply's fits by form name (the reply lists every form that fit)."""
    return {fit["form"]: fit for fit in payload["fits"]}


def test_linear_fit_reply_shape():
    # y = 2x + 1 exactly: in rational mode the linear / quadratic / cubic forms all fit it
    # exactly (the power form is dropped — its logs are irrational). Each fitted form is one
    # entry in the reply; this checks the linear entry's shape and exact-verdict fields.
    payload = _fit([1, 2, 3, 4], [3, 5, 7, 9], mode="rational")
    assert payload["error"] is None
    assert payload["mode"] == "rational"
    by_form = _by_form(payload)
    assert "power" not in by_form  # rational refuses the power form's logs
    fit = by_form["linear"]
    assert fit["equation"] == "2*x + 1"
    assert [p["name"] for p in fit["parameters"]] == ["a", "b"]
    assert fit["parameters"][0]["value"] == "2 (exact)"
    assert fit["parameters"][1]["value"] == "1 (exact)"
    assert fit["fit_error"] == "0 (exact)"
    assert fit["exact"] is True
    # The polynomial forms recover the same line exactly (leading coefficients zero).
    assert by_form["quadratic"]["fit_error"] == "0 (exact)"
    assert by_form["cubic"]["fit_error"] == "0 (exact)"


def test_nonlinear_data_is_fit_by_the_matching_form():
    # y = 2x**1.5 over the wire in floating-point: the power form recovers a≈2, b≈1.5
    # with a near-zero error, alongside the polynomial approximations.
    ys = [2 * x**1.5 for x in (1, 2, 3, 4, 5)]
    payload = _fit([1, 2, 3, 4, 5], ys, mode="floating-point")
    power = _by_form(payload)["power"]
    a, b = power["parameters"]
    assert float(a["value"].split()[0]) == pytest.approx(2.0, rel=1e-6)
    assert float(b["value"].split()[0]) == pytest.approx(1.5, rel=1e-6)
    assert power["exact"] is False


def test_default_mode_is_fixed_point_with_a_sub_unit_floor():
    # Omitting mode/floor: fixed-point at the default floor 9, so the parameters keep nine
    # fractional digits rather than rounding to whole numbers. y = 1.5x + 0.25 is exactly
    # linear, so the line is among the best fits and shows its sub-unit coefficients.
    payload = _fit([1, 2, 3, 4], [1.75, 3.25, 4.75, 6.25])
    assert payload["mode"] == "fixed-point"
    fit = _by_form(payload)["linear"]
    # slope 1.5, intercept 0.25 at scale 9 — nine fractional digits, not rounded to integers.
    assert fit["parameters"][0]["value"].startswith("1.500000000")
    assert fit["parameters"][1]["value"].startswith("0.250000000")
    assert fit["equation"].startswith("1.500000000*x + 0.250000000")


def test_reply_carries_a_top_level_precision_verdict_for_the_best_fit():
    # 44.6: like solver, the reply's top-level exact/precision describe the BEST (least-
    # error) fit's error. y = 2x + 1 in rational fits exactly, so the headline is exact.
    payload = _fit([1, 2, 3, 4], [3, 5, 7, 9], mode="rational")
    assert payload["exact"] is True
    # The headline mirrors the first (best-ranked) fit's own verdict.
    assert payload["exact"] == payload["fits"][0]["exact"]
    assert payload["precision"] == payload["fits"][0]["precision"]
    # Floating-point conservatively flags every result inexact, headline included.
    noisy = _fit([1, 1.5, 2.0], [2, 5.8, 8.9], mode="floating-point")
    assert noisy["exact"] is False
    assert noisy["exact"] == noisy["fits"][0]["exact"]


def test_only_the_best_three_forms_come_back():
    # 44.5: y = 2x**1.5 in floating-point fits all four forms, but the reply lists only
    # the best three by error — the worst (linear) is dropped, the matching power leads.
    ys = [2 * x**1.5 for x in (1, 2, 3, 4, 5)]
    payload = _fit([1, 2, 3, 4, 5], ys, mode="floating-point")
    forms = [fit["form"] for fit in payload["fits"]]
    assert len(forms) == 3
    assert forms[0] == "power" and "linear" not in forms


def test_reciprocal_form_fits_exactly_over_the_wire():
    # 44.2.8: y = 3 + 2/x in rational mode — the reciprocal form recovers a=2, b=3 exactly,
    # rendered as a pasteable "2/x + 3" with an exact error.
    payload = _fit([1, 2, 4], [5, 4, 3.5], mode="rational")
    assert payload["error"] is None
    fit = _by_form(payload)["reciprocal"]
    assert fit["equation"] == "2/x + 3"
    assert fit["parameters"][0]["value"] == "2 (exact)"
    assert fit["parameters"][1]["value"] == "3 (exact)"
    assert fit["fit_error"] == "0 (exact)"
    assert fit["exact"] is True


def test_exponential_form_fits_over_the_wire():
    # 44.2.5: y = 3*exp(0.5*x) in floating-point — the exponential form recovers a≈3, b≈0.5
    # and renders an exp(...) equation pasteable into calculate.
    xs = [0, 1, 2, 3, 4]
    ys = [3 * math.exp(0.5 * x) for x in xs]
    payload = _fit(xs, ys, mode="floating-point")
    fit = _by_form(payload)["exponential"]
    a, b = fit["parameters"]
    assert float(a["value"].split()[0]) == pytest.approx(3.0, rel=1e-6)
    assert float(b["value"].split()[0]) == pytest.approx(0.5, rel=1e-6)
    assert "exp(" in fit["equation"]
    assert fit["exact"] is False


def test_sinusoidal_form_fits_over_the_wire():
    # 44.2.9: y = 3*sin(2x + 0.5) + 1 in floating-point — the iterative frequency search
    # recovers the frequency b≈2, the positive amplitude a≈3 and offset d≈1 with a small
    # error, rendered as a pasteable a*sin(b*x + c) + d.
    xs = [i * 0.25 for i in range(24)]
    ys = [3 * math.sin(2 * x + 0.5) + 1 for x in xs]
    payload = _fit(xs, ys, mode="floating-point")
    fit = _by_form(payload)["sinusoidal"]
    params = {p["name"]: float(p["value"].split()[0]) for p in fit["parameters"]}
    assert params["a"] == pytest.approx(3.0, rel=1e-6) and params["a"] > 0
    assert params["b"] == pytest.approx(2.0, rel=1e-6)
    assert params["d"] == pytest.approx(1.0, rel=1e-6)
    assert "sin(" in fit["equation"]
    assert float(fit["fit_error"].split()[0]) == pytest.approx(0.0, abs=1e-9)


def test_gaussian_form_fits_over_the_wire():
    # 44.2.10: y = 2*exp(-(x-3)**2/(2*1**2)) in floating-point — Caruana's closed-form fit
    # recovers the peak a≈2, centre b≈3 and width c≈1 with a small error, rendered as a
    # pasteable a*exp(-(x-b)**2/(2*c**2)).
    xs = [i * 0.5 for i in range(13)]
    ys = [2 * math.exp(-((x - 3) ** 2) / (2 * 1**2)) for x in xs]
    payload = _fit(xs, ys, mode="floating-point")
    fit = _by_form(payload)["gaussian"]
    params = {p["name"]: float(p["value"].split()[0]) for p in fit["parameters"]}
    assert params["a"] == pytest.approx(2.0, rel=1e-6)
    assert params["b"] == pytest.approx(3.0, rel=1e-6)
    assert params["c"] == pytest.approx(1.0, rel=1e-6)
    assert "exp(-(" in fit["equation"]
    assert float(fit["fit_error"].split()[0]) == pytest.approx(0.0, abs=1e-9)


def test_saturation_form_fits_exactly_over_the_wire():
    # 44.2.11: y = x/(x + 2) in rational mode — the saturation form recovers a=1, b=2 exactly
    # via the double-reciprocal line, rendered as a pasteable "x/(1*x + 2)" with an exact error.
    payload = _fit([2, 0.5, 8], [0.5, 0.2, 0.8], mode="rational")
    assert payload["error"] is None
    fit = _by_form(payload)["saturation"]
    assert fit["equation"] == "x/(1*x + 2)"
    assert fit["parameters"][0]["value"] == "1 (exact)"
    assert fit["parameters"][1]["value"] == "2 (exact)"
    assert fit["fit_error"] == "0 (exact)"
    assert fit["exact"] is True


def test_hyperbolic_form_fits_exactly_over_the_wire():
    # 44.2.12: y = 1/(x + 1) in rational mode — the hyperbolic form recovers a=1, b=1 exactly
    # via the line 1/y = a*x + b, rendered as a pasteable "1/(1*x + 1)" with an exact error.
    payload = _fit([1, 3, 4, 9], [0.5, 0.25, 0.2, 0.1], mode="rational")
    assert payload["error"] is None
    fit = _by_form(payload)["hyperbolic"]
    assert fit["equation"] == "1/(1*x + 1)"
    assert fit["parameters"][0]["value"] == "1 (exact)"
    assert fit["parameters"][1]["value"] == "1 (exact)"
    assert fit["fit_error"] == "0 (exact)"
    assert fit["exact"] is True


def test_laurent_form_fits_exactly_over_the_wire():
    # 44.2.13: y = 2 + 3*x + 1/x in rational mode (x powers of two so 1/x is exact in binary)
    # — the Laurent form recovers a=2, b=3, c=1 exactly via the direct basis {1, x, 1/x},
    # rendered as a pasteable "2 + 3*x + 1/x" with an exact error.
    payload = _fit([1, 2, 4, 8], [6, 8.5, 14.25, 26.125], mode="rational")
    assert payload["error"] is None
    fit = _by_form(payload)["laurent"]
    assert fit["equation"] == "2 + 3*x + 1/x"
    assert [p["value"] for p in fit["parameters"]] == ["2 (exact)", "3 (exact)", "1 (exact)"]
    assert fit["fit_error"] == "0 (exact)"
    assert fit["exact"] is True


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
