# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit-level coverage of the fit module's engine (TODO 44).

The companion to test_fit_e2e.py: where that drives the whole `curve_fit` TOOL seam
(request dict -> dispatch -> fit -> reply), this exercises the building blocks in
isolation — the closed-form least-squares fitters (linear, the polynomial normal
equations, the log-linearised power law) and their type-faithful behaviour across modes,
the curve registry, and the drop-and-continue on data a form cannot fit.
"""

import pytest

from mcp_abacus.expr.value import Mode
from mcp_abacus.fit import CURVE_FORMS, FitError, FitResult, fit_all


def _pick(results: list[FitResult], form: str) -> FitResult:
    """The one fitted result for `form`, or fail the test if it was dropped."""
    matches = [r for r in results if r.form == form]
    assert matches, f"the {form} form was not among the fits"
    return matches[0]


def test_perfect_line_is_recovered_exactly_in_rational():
    # y = 2x + 1 exactly — rational least squares is exact, slope/intercept on the nose.
    result = _pick(fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.RATIONAL, 0), "linear")
    params = dict(result.parameters)
    assert params["a"].to_string() == "2"
    assert params["b"].to_string() == "1"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_perfect_line_is_exact_in_fixed_point():
    # Fixed-point at a non-zero floor holds the integer slope/intercept exactly.
    result = _pick(fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.FIXED_POINT, 9), "linear")
    params = dict(result.parameters)
    assert params["a"].to_float() == 2.0
    assert params["b"].to_float() == 1.0
    assert result.error.to_float() == 0.0


def test_noisy_data_gives_the_least_squares_line_in_rational():
    # The prompt's example: x=[1,1.5,2], y=[2,5.8,8.9]. OLS gives slope 69/10,
    # intercept -287/60, SSR 49/600 — all exact in rational mode.
    result = _pick(fit_all([1, 1.5, 2.0], [2, 5.8, 8.9], Mode.RATIONAL, 0), "linear")
    params = dict(result.parameters)
    assert params["a"].to_string() == "69/10"
    assert params["b"].to_string() == "-287/60"
    assert result.error.to_string() == "49/600"


def test_floating_point_fit_is_close_but_inexact():
    result = _pick(fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.FLOATING_POINT, 0), "linear")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0)
    assert params["b"].to_float() == pytest.approx(1.0)
    # Floating-point conservatively flags every result inexact.
    assert not result.error.exact


def test_equation_renders_over_x_with_a_clean_sign():
    # A positive intercept reads "+ b"; the noisy case's negative intercept reads "- |b|".
    positive = _pick(fit_all([1, 2, 3], [3, 5, 7], Mode.RATIONAL, 0), "linear")
    assert positive.equation == "2*x + 1"
    negative = _pick(fit_all([1, 1.5, 2.0], [2, 5.8, 8.9], Mode.RATIONAL, 0), "linear")
    assert negative.equation == "69/10*x - 287/60"


def test_quadratic_recovers_exact_parabola_in_rational():
    # y = 2x**2 + 3x + 1 exactly — a parabola is linear in its coefficients, so the
    # normal-equations solve is exact in rational mode (error 0, on the nose).
    result = _pick(fit_all([1, 2, 3, 4, 5], [6, 15, 28, 45, 66], Mode.RATIONAL, 0), "quadratic")
    params = dict(result.parameters)
    assert params["a"].to_string() == "2"
    assert params["b"].to_string() == "3"
    assert params["c"].to_string() == "1"
    assert result.equation == "2*x**2 + 3*x + 1"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_cubic_recovers_exact_cubic_in_rational():
    # y = x**3 - 2x**2 + 1 exactly — recovered with error 0 in rational mode. The leading
    # coefficient is 1 and the x term vanishes, so the equation shows the honest "0*x".
    result = _pick(
        fit_all([0, 1, 2, 3, 4, 5], [1, 0, 1, 10, 33, 76], Mode.RATIONAL, 0), "cubic"
    )
    params = dict(result.parameters)
    assert params["a"].to_string() == "1"
    assert params["b"].to_string() == "-2"
    assert params["c"].to_string() == "0"
    assert params["d"].to_string() == "1"
    assert result.equation == "1*x**3 - 2*x**2 + 0*x + 1"
    assert result.error.to_string() == "0"


def test_polynomial_drops_when_data_underdetermines_it():
    # Three points cannot determine a cubic's four coefficients (a singular normal-equations
    # matrix), so the cubic is DROPPED — but the lower-order forms still fit and come back.
    forms = [r.form for r in fit_all([1, 2, 3], [2, 5, 10], Mode.RATIONAL, 0)]
    assert "cubic" not in forms
    assert "linear" in forms and "quadratic" in forms


def test_power_law_fit_in_floating_point():
    # y = 2*x**1.5 — the power form linearises under logs and recovers a≈2, b≈1.5.
    ys = [2 * x**1.5 for x in (1, 2, 3, 4, 5)]
    result = _pick(fit_all([1, 2, 3, 4, 5], ys, Mode.FLOATING_POINT, 0), "power")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(1.5, rel=1e-6)
    assert not result.error.exact


def test_power_is_dropped_in_rational_mode():
    # The log-linearisation needs ln/exp, which are irrational — rational mode refuses
    # them (exact-or-refuse), so the power form is dropped while the polynomials remain.
    forms = [r.form for r in fit_all([1, 2, 3, 4], [2, 8, 18, 32], Mode.RATIONAL, 0)]
    assert "power" not in forms
    assert "linear" in forms


def test_power_is_dropped_when_x_is_not_positive():
    # The power law needs x > 0 (and y > 0) for the logs; a non-positive x drops it, but
    # the polynomial forms, which have no such restriction, still fit.
    forms = [r.form for r in fit_all([-1, 1, 2, 3], [1, 2, 4, 8], Mode.FLOATING_POINT, 0)]
    assert "power" not in forms
    assert "quadratic" in forms


def test_fits_are_ranked_by_error_best_first():
    # Curved data: the higher-order / matching forms fit better, so the results come back
    # ordered by ascending fit error (44.5) — the least-error fit leads. The power form,
    # which matches y = 2x**1.5, is the best fit; linear (the worst) is truncated away.
    ys = [2 * x**1.5 for x in (1, 2, 3, 4, 5)]
    results = fit_all([1, 2, 3, 4, 5], ys, Mode.FLOATING_POINT, 0)
    errors = [r.error.to_float() for r in results]
    assert errors == sorted(errors)
    assert results[0].form == "power"


def test_only_the_best_three_forms_are_returned():
    # 44.5: y = 2x**1.5 fits all four forms (power exactly, the polynomials approximately),
    # but fit_all returns only the best three by error — the worst (linear) is truncated.
    ys = [2 * x**1.5 for x in (1, 2, 3, 4, 5)]
    results = fit_all([1, 2, 3, 4, 5], ys, Mode.FLOATING_POINT, 0)
    assert len(results) == 3
    forms = [r.form for r in results]
    assert "power" in forms and "linear" not in forms  # the matching form leads, linear trails off


def test_fewer_than_three_forms_when_fewer_fit():
    # 44.5: with only three points a cubic is underdetermined and dropped, so only the
    # linear and quadratic forms fit — fewer than three come back, not padded.
    results = fit_all([1, 2, 3], [2, 5, 10], Mode.RATIONAL, 0)
    assert [r.form for r in results] == ["quadratic", "linear"]


def test_exact_fits_tie_break_to_the_simplest_form():
    # y = 2x + 1: linear, quadratic and cubic all fit exactly (error 0); a stable sort on
    # the tie keeps registry (ascending-complexity) order, so linear leads.
    results = fit_all([1, 2, 3, 4], [3, 5, 7, 9], Mode.RATIONAL, 0)
    exact = [r.form for r in results if r.error.to_float() == 0.0]
    assert exact[:3] == ["linear", "quadratic", "cubic"]


def test_no_form_fitting_surfaces_the_linear_reason():
    # Every x equal -> no form fits; fit_all re-raises the first (linear) reason.
    with pytest.raises(FitError) as excinfo:
        fit_all([2, 2, 2], [1, 2, 3], Mode.RATIONAL, 0)
    assert "every x value is equal" in excinfo.value.message.lower()


def test_curve_library_declares_every_form_with_its_metadata():
    # 44.2.1-44.2.4: the registry declares linear plus the quadratic/cubic/power forms,
    # each with its parameter names, model template and domain limit.
    by_name = {form.name: form for form in CURVE_FORMS}
    assert list(by_name) == ["linear", "quadratic", "cubic", "power"]
    assert by_name["linear"].parameters == ("a", "b")
    assert by_name["quadratic"].parameters == ("a", "b", "c")
    assert by_name["quadratic"].template == "a*x**2 + b*x + c"
    assert by_name["cubic"].parameters == ("a", "b", "c", "d")
    assert by_name["cubic"].template == "a*x**3 + b*x**2 + c*x + d"
    # Only the power form restricts x; the polynomial forms accept any x.
    assert by_name["power"].domain == "x > 0"
    assert by_name["linear"].domain is None


def test_every_form_is_now_wired_to_a_fitter():
    # 44.3: all four declared forms carry a live closed-form fitter.
    assert all(form.fit is not None for form in CURVE_FORMS)
