# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit-level coverage of the fit module's engine (TODO 44).

The companion to test_fit_e2e.py: where that drives the whole `curve_fit` TOOL seam
(request dict -> dispatch -> fit -> reply), this exercises the building blocks in
isolation — the linear least-squares fit and its type-faithful behaviour across
modes, the curve registry, and the degenerate-data refusal.
"""

import pytest

from mcp_abacus.expr.value import Mode
from mcp_abacus.fit import CURVE_FORMS, FitError, fit_all


def test_perfect_line_is_recovered_exactly_in_rational():
    # y = 2x + 1 exactly — rational least squares is exact, slope/intercept on the nose.
    (result,) = fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.RATIONAL, 0)
    assert result.form == "linear"
    params = dict(result.parameters)
    assert params["a"].to_string() == "2"
    assert params["b"].to_string() == "1"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_perfect_line_is_exact_in_fixed_point():
    # Fixed-point at a non-zero floor holds the integer slope/intercept exactly.
    (result,) = fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.FIXED_POINT, 9)
    params = dict(result.parameters)
    assert params["a"].to_float() == 2.0
    assert params["b"].to_float() == 1.0
    assert result.error.to_float() == 0.0


def test_noisy_data_gives_the_least_squares_line_in_rational():
    # The prompt's example: x=[1,1.5,2], y=[2,5.8,8.9]. OLS gives slope 69/10,
    # intercept -287/60, SSR 49/600 — all exact in rational mode.
    (result,) = fit_all([1, 1.5, 2.0], [2, 5.8, 8.9], Mode.RATIONAL, 0)
    params = dict(result.parameters)
    assert params["a"].to_string() == "69/10"
    assert params["b"].to_string() == "-287/60"
    assert result.error.to_string() == "49/600"


def test_floating_point_fit_is_close_but_inexact():
    (result,) = fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.FLOATING_POINT, 0)
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0)
    assert params["b"].to_float() == pytest.approx(1.0)
    # Floating-point conservatively flags every result inexact.
    assert not result.error.exact


def test_equation_renders_over_x_with_a_clean_sign():
    # A positive intercept reads "+ b"; the noisy case's negative intercept reads "- |b|".
    (positive,) = fit_all([1, 2, 3], [3, 5, 7], Mode.RATIONAL, 0)
    assert positive.equation == "2*x + 1"
    (negative,) = fit_all([1, 1.5, 2.0], [2, 5.8, 8.9], Mode.RATIONAL, 0)
    assert negative.equation == "69/10*x - 287/60"


def test_vertical_line_is_refused():
    # Every x equal -> the slope is undefined (zero denominator), a FitError.
    with pytest.raises(FitError) as excinfo:
        fit_all([2, 2, 2], [1, 2, 3], Mode.RATIONAL, 0)
    assert "every x value is equal" in excinfo.value.message.lower()


def test_curve_library_holds_the_linear_form():
    # 44.2.1: the registry carries the one wired form today.
    names = [form.name for form in CURVE_FORMS]
    assert names == ["linear"]
    assert dict.fromkeys(CURVE_FORMS[0].parameters) == {"a": None, "b": None}
