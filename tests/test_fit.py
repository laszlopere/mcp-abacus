# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit-level coverage of the fit module's engine (TODO 44).

The companion to test_fit_e2e.py: where that drives the whole `curve_fit` TOOL seam
(request dict -> dispatch -> fit -> reply), this exercises the building blocks in
isolation — the closed-form least-squares fitters (linear, the polynomial normal
equations, the log-linearised power law) and their type-faithful behaviour across modes,
the curve registry, and the drop-and-continue on data a form cannot fit.
"""

import math

import pytest

from mcp_abacus.expr.value import Mode, Value
from mcp_abacus.fit import (
    CURVE_FORMS,
    FitError,
    FitResult,
    _fit_scale,
    _linear_basis_fitter,
    fit_all,
)


def _pick(results: list[FitResult], form: str) -> FitResult:
    """The one fitted result for `form`, or fail the test if it was dropped."""
    matches = [r for r in results if r.form == form]
    assert matches, f"the {form} form was not among the fits"
    return matches[0]


def _fit_one(form: str, xs: list[float], ys: list[float], mode: Mode, floor: int = 0) -> FitResult:
    """Fit a SINGLE named form directly, bypassing fit_all's best-3 truncation.

    Materialises the data exactly as fit_all does (same scale) and calls that form's fitter, so
    the result is identical to fit_all's — but visible even when the form ranks outside the best
    three (a low-parameter form like linear is beaten on few points by any exactly-fitting form).
    """
    scale = _fit_scale(mode, floor)
    x_values = [Value.from_real(x, mode, scale) for x in xs]
    y_values = [Value.from_real(y, mode, scale) for y in ys]
    fitter = {f.name: f for f in CURVE_FORMS}[form].fit
    assert fitter is not None
    return fitter(x_values, y_values, mode, scale)


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
    # intercept -287/60, SSR 49/600 — all exact in rational mode. Fit linear directly: on three
    # points any exactly-fitting 3-parameter form outranks it, so it need not be in the best three.
    result = _fit_one("linear", [1, 1.5, 2.0], [2, 5.8, 8.9], Mode.RATIONAL)
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
    # A positive intercept reads "+ b"; the noisy case's negative intercept reads "- |b|". Fit
    # linear directly — on three points it is outranked by any exactly-fitting 3-parameter form.
    positive = _fit_one("linear", [1, 2, 3], [3, 5, 7], Mode.RATIONAL)
    assert positive.equation == "2*x + 1"
    negative = _fit_one("linear", [1, 1.5, 2.0], [2, 5.8, 8.9], Mode.RATIONAL)
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
    result = _pick(fit_all([0, 1, 2, 3, 4, 5], [1, 0, 1, 10, 33, 76], Mode.RATIONAL, 0), "cubic")
    params = dict(result.parameters)
    assert params["a"].to_string() == "1"
    assert params["b"].to_string() == "-2"
    assert params["c"].to_string() == "0"
    assert params["d"].to_string() == "1"
    assert result.equation == "1*x**3 - 2*x**2 + 0*x + 1"
    assert result.error.to_string() == "0"


def test_polynomial_drops_when_data_underdetermines_it():
    # Three points cannot determine a cubic's four coefficients (a singular normal-equations
    # matrix), so the cubic is DROPPED — but the lower-order forms still fit and come back
    # (quadratic among the best three; the best-3 truncation may trail off linear).
    forms = [r.form for r in fit_all([1, 2, 3], [2, 5, 10], Mode.RATIONAL, 0)]
    assert "cubic" not in forms
    assert "quadratic" in forms


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
    assert "quadratic" in forms  # the polynomials still fit (best-3 may trail off linear)


def test_power_is_dropped_when_x_is_not_positive():
    # The power law needs x > 0 (and y > 0) for the logs; a non-positive x drops it, but
    # the polynomial forms, which have no such restriction, still fit.
    forms = [r.form for r in fit_all([-1, 1, 2, 3], [1, 2, 4, 8], Mode.FLOATING_POINT, 0)]
    assert "power" not in forms
    assert "quadratic" in forms


def test_exponential_fit_in_floating_point():
    # 44.2.5: y = 3*exp(0.5*x) — the exponential linearises under logs of y alone (x raw),
    # so it recovers a≈3, b≈0.5 in floating-point.
    xs = [0, 1, 2, 3, 4]
    ys = [3 * math.exp(0.5 * x) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "exponential")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(3.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(0.5, rel=1e-6)
    assert "*exp(" in result.equation and "*x)" in result.equation
    assert not result.error.exact


def test_exponential_is_dropped_in_rational_mode():
    # The exponential needs ln/exp, which are irrational — rational mode refuses them, so
    # the form is dropped while the polynomials remain (like the power form).
    ys = [3 * math.exp(0.5 * x) for x in (0, 1, 2, 3, 4)]
    forms = [r.form for r in fit_all([0, 1, 2, 3, 4], ys, Mode.RATIONAL, 0)]
    assert "exponential" not in forms
    assert "quadratic" in forms  # the polynomials remain (best-3 may trail off linear)


def test_exp_reciprocal_fit_in_floating_point():
    # 44.2.14: y = 3*exp(2/x) (the Arrhenius law) — it linearises under logs of y against 1/x,
    # so it recovers a≈3, b≈2 in floating-point. The equation reads a*exp(b/x), not a*exp(b*x).
    xs = [0.5, 1, 2, 3, 4]
    ys = [3 * math.exp(2 / x) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "exp-reciprocal")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(3.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert "*exp(" in result.equation and "/x)" in result.equation
    assert not result.error.exact


def test_exp_reciprocal_is_dropped_in_rational_mode():
    # Like the exponential/power forms, its log-linearisation needs ln/exp — irrational, so
    # rational mode refuses them and the form drops while the polynomials remain.
    ys = [3 * math.exp(2 / x) for x in (0.5, 1, 2, 3, 4)]
    forms = [r.form for r in fit_all([0.5, 1, 2, 3, 4], ys, Mode.RATIONAL, 0)]
    assert "exp-reciprocal" not in forms
    assert "cubic" in forms  # a polynomial remains (best-3 may trail off quadratic)


def test_exp_reciprocal_is_dropped_when_some_x_is_zero():
    # a*exp(b/x) needs the reciprocal 1/x; an x = 0 datum raises ZeroDivisionError, which the
    # fitter catches as a clean DROP (no crash) while the forms with no 1/x still fit.
    ys = [3 * math.exp(2 / x) for x in (1, 2, 3, 4)]
    forms = [r.form for r in fit_all([0, 1, 2, 3], ys, Mode.FLOATING_POINT, 0)]
    assert "exp-reciprocal" not in forms
    assert "quadratic" in forms


def test_logarithmic_fit_in_floating_point():
    # 44.2.6: y = 2 + 3*ln(x) — affine in {1, ln x}, recovered via the shared basis fitter.
    xs = [1, 2, 3, 4, 5]
    ys = [2 + 3 * math.log(x) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "logarithmic")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(3.0, rel=1e-6)
    assert "*ln(x)" in result.equation
    assert not result.error.exact


def test_logarithmic_is_dropped_in_rational_mode():
    # ln x is irrational, so rational mode refuses the basis and drops the logarithmic form.
    ys = [2 + 3 * math.log(x) for x in (1, 2, 3, 4, 5)]
    forms = [r.form for r in fit_all([1, 2, 3, 4, 5], ys, Mode.RATIONAL, 0)]
    assert "logarithmic" not in forms
    assert "quadratic" in forms  # the polynomial forms have no such restriction and remain


def test_square_root_fit_in_floating_point():
    # 44.2.7: y = 2*sqrt(x) + 1 — affine in {sqrt x, 1}; on perfect-square x the data is
    # integral (y = [3, 5, 7, 9]) and the fit recovers a≈2, b≈1.
    xs = [1, 4, 9, 16]
    ys = [2 * math.sqrt(x) + 1 for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "square-root")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(1.0, rel=1e-6)
    assert "*sqrt(x)" in result.equation
    assert not result.error.exact


def test_square_root_is_dropped_in_rational_mode():
    # sqrt x is irrational for non-square x, so rational mode refuses it and drops the form.
    ys = [2 * math.sqrt(x) + 1 for x in (1, 2, 3, 5)]
    forms = [r.form for r in fit_all([1, 2, 3, 5], ys, Mode.RATIONAL, 0)]
    assert "square-root" not in forms
    assert "quadratic" in forms  # the polynomials still fit (best-3 may trail off linear)


def test_reciprocal_recovers_exactly_in_rational():
    # 44.2.8: y = 3 + 2/x — the reciprocal basis {1/x, 1} is exact in rational mode (1/x of
    # a rational is rational), so a=2, b=3 are recovered on the nose with error 0.
    result = _pick(fit_all([1, 2, 4], [5, 4, 3.5], Mode.RATIONAL, 0), "reciprocal")
    params = dict(result.parameters)
    assert params["a"].to_string() == "2"
    assert params["b"].to_string() == "3"
    assert result.equation == "2/x + 3"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_reciprocal_renders_a_clean_negative_constant():
    # The affine renderer splits the sign: a negative constant reads "- |b|", not "+ -b".
    result = _pick(fit_all([1, 2, 4], [-1, -2, -2.5], Mode.RATIONAL, 0), "reciprocal")
    assert result.equation == "2/x - 3"  # y = 2/x - 3


def test_reciprocal_is_dropped_when_some_x_is_zero():
    # 1/x at x = 0 raises ZeroDivisionError (NOT NotRepresentableError); the basis fitter
    # catches it as a clean DROP, not a crash — the other forms still fit and come back.
    forms = [r.form for r in fit_all([0, 1, 2, 3], [1, 2, 3, 4], Mode.RATIONAL, 0)]
    assert "reciprocal" not in forms
    assert "linear" in forms and "quadratic" in forms


def test_sinusoidal_recovers_frequency_amplitude_and_offset_in_floating_point():
    # 44.2.9: y = 3*sin(2x + 0.5) + 1 over well-spaced x — the iterative frequency search
    # recovers the frequency b≈2, the positive amplitude a≈3 and the offset d≈1 with a
    # near-zero residual. The raw phase c is left unasserted (a→−a with c→c+π is the same
    # curve, so it wraps); the canonical positive amplitude and the frequency/offset are the
    # stable handles.
    xs = [i * 0.25 for i in range(24)]  # 24 well-spaced points over [0, 6) pin the frequency
    ys = [3 * math.sin(2 * x + 0.5) + 1 for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "sinusoidal")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(3.0, rel=1e-6)
    assert params["a"].to_float() > 0  # canonical positive amplitude (hypot)
    assert params["b"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["d"].to_float() == pytest.approx(1.0, rel=1e-6)
    assert result.error.to_float() == pytest.approx(0.0, abs=1e-9)  # a near-perfect fit
    assert "*sin(" in result.equation and "*x" in result.equation


def test_sinusoidal_is_dropped_in_rational_mode():
    # sin of a non-zero rational is irrational, so rational mode refuses the form (its only
    # mode drop — there is no x domain limit); the polynomial forms have no such limit and remain.
    xs = [i * 0.25 for i in range(24)]
    ys = [3 * math.sin(2 * x + 0.5) + 1 for x in xs]
    forms = [r.form for r in fit_all(xs, ys, Mode.RATIONAL, 0)]
    assert "sinusoidal" not in forms
    assert "cubic" in forms  # the polynomial forms still fit


def test_sinusoidal_is_dropped_when_too_few_distinct_x():
    # The sinusoid's four parameters need at least four distinct x to be determined; three
    # distinct points drop it cleanly (no crash) while the lower-order forms still fit.
    forms = [r.form for r in fit_all([1, 2, 3], [2, 5, 10], Mode.FLOATING_POINT, 0)]
    assert "sinusoidal" not in forms
    assert "cubic" in forms  # a lower-order form still fits the three points


def test_gaussian_recovers_peak_centre_and_width_in_floating_point():
    # 44.2.10: y = 2*exp(-(x-3)**2/(2*1**2)) over well-spaced x around the peak — Caruana's
    # method logs the data to a quadratic and recovers the peak a≈2, centre b≈3 and width
    # c≈1 with a near-zero residual.
    xs = [i * 0.5 for i in range(13)]  # 0, 0.5, …, 6 — well-spaced around the peak at 3
    ys = [2 * math.exp(-((x - 3) ** 2) / (2 * 1**2)) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "gaussian")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(3.0, rel=1e-6)
    assert params["c"].to_float() == pytest.approx(1.0, rel=1e-6)
    assert params["c"].to_float() > 0  # the positive width
    assert result.error.to_float() == pytest.approx(0.0, abs=1e-9)  # a near-perfect fit
    assert "*exp(-(x - " in result.equation  # the pasteable bell, clean positive-centre sign


def test_gaussian_is_dropped_in_rational_mode():
    # The Caruana fit needs ln/exp/sqrt, which are irrational — rational mode refuses them
    # (exact-or-refuse), so the gaussian is dropped while the polynomials remain.
    xs = [i * 0.5 for i in range(13)]
    ys = [2 * math.exp(-((x - 3) ** 2) / (2 * 1**2)) for x in xs]
    forms = [r.form for r in fit_all(xs, ys, Mode.RATIONAL, 0)]
    assert "gaussian" not in forms
    assert "quadratic" in forms


def test_gaussian_is_dropped_when_data_is_not_a_downward_parabola():
    # ln y = x**2 is an UPWARD parabola in the logs (leading coefficient p2 ≈ 1 ≥ 0), so the
    # bell cannot be recovered (a downward parabola is required) — the gaussian drops cleanly
    # while the polynomial forms still fit the data.
    xs = [-2, -1, 0, 1, 2, 3]
    ys = [math.exp(x**2) for x in xs]  # log-quadratic opens upward, p2 > 0
    forms = [r.form for r in fit_all(xs, ys, Mode.FLOATING_POINT, 0)]
    assert "gaussian" not in forms
    assert "quadratic" in forms


def test_gaussian_is_dropped_when_some_y_is_not_positive():
    # A non-positive y has no real log, so the Caruana linearisation is undefined and the
    # gaussian drops cleanly — the polynomial forms, which need no log, still fit.
    xs = [0, 1, 2, 3, 4]
    ys = [1.0, 2.0, 0.0, 2.0, 1.0]  # a zero y kills the log
    forms = [r.form for r in fit_all(xs, ys, Mode.FLOATING_POINT, 0)]
    assert "gaussian" not in forms
    assert "linear" in forms  # the log-free polynomial/affine forms still fit


def test_hoerl_recovers_base_and_exponent_in_floating_point():
    # 44.2.15: y = 2*1.5**x*x**0.5 — the Hoerl model logs to a line over {1, x, ln x}, so it
    # recovers a≈2, b≈1.5, c≈0.5 in floating-point. The equation reads a*b**x*x**c.
    xs = [1, 2, 3, 4, 5]
    ys = [2 * 1.5**x * x**0.5 for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "hoerl")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(1.5, rel=1e-6)
    assert params["c"].to_float() == pytest.approx(0.5, rel=1e-6)
    assert "**x*x**" in result.equation  # a*b**x*x**c
    assert not result.error.exact


def test_hoerl_is_dropped_in_rational_mode():
    # The log-linearisation needs ln/exp, which are irrational — rational mode refuses them, so
    # the Hoerl form drops while the polynomials remain (like the power/gaussian forms).
    ys = [2 * 1.5**x * x**0.5 for x in (1, 2, 3, 4, 5)]
    forms = [r.form for r in fit_all([1, 2, 3, 4, 5], ys, Mode.RATIONAL, 0)]
    assert "hoerl" not in forms
    assert "cubic" in forms  # a polynomial remains (best-3 may trail off quadratic)


def test_hoerl_is_dropped_when_some_x_is_not_positive():
    # a*b**x*x**c needs ln x, so a non-positive x has no real log and the form drops cleanly —
    # the polynomial forms, which need no log, still fit.
    ys = [2 * 1.5 ** abs(x) * abs(x) ** 0.5 + 1 for x in (-1, 1, 2, 3)]
    forms = [r.form for r in fit_all([-1, 1, 2, 3], ys, Mode.FLOATING_POINT, 0)]
    assert "hoerl" not in forms
    assert "quadratic" in forms


def test_weibull_recovers_scale_and_shape_in_floating_point():
    # 44.2.16: y = 1 - exp(-(x/2)**1.5) — the Weibull plot ln(-ln(1-y)) = b*ln x - b*ln a is a
    # line, so it recovers the scale a≈2 and shape b≈1.5 in floating-point, error ≈ 0.
    xs = [1, 2, 3, 4, 5]
    ys = [1 - math.exp(-((x / 2) ** 1.5)) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "weibull")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)  # the scale L
    assert params["b"].to_float() == pytest.approx(1.5, rel=1e-6)  # the shape k
    assert result.error.to_float() == pytest.approx(0.0, abs=1e-9)
    assert "exp(-(x/" in result.equation and ")**" in result.equation
    assert not result.error.exact


def test_weibull_is_dropped_in_rational_mode():
    # The double-log transform needs ln/exp, which are irrational — rational mode refuses them,
    # so the Weibull form drops while the polynomials remain (like the power/gaussian forms).
    ys = [1 - math.exp(-((x / 2) ** 1.5)) for x in (1, 2, 3, 4, 5)]
    forms = [r.form for r in fit_all([1, 2, 3, 4, 5], ys, Mode.RATIONAL, 0)]
    assert "weibull" not in forms
    assert "cubic" in forms  # a polynomial remains


def test_weibull_is_dropped_when_y_is_outside_the_unit_interval():
    # The ordinate ln(-ln(1 - y)) needs 0 < y < 1; a y = 1 makes 1 - y = 0, so the inner log is
    # undefined and the form drops cleanly while the log-free forms still fit.
    forms = [r.form for r in fit_all([1, 2, 3, 4], [0.3, 0.6, 0.8, 1.0], Mode.FLOATING_POINT, 0)]
    assert "weibull" not in forms
    assert "cubic" in forms


def test_weibull_is_dropped_when_data_is_not_weibull_shaped():
    # A valid Weibull CDF rises, so its Weibull-plot slope (the shape b) is positive; decreasing
    # data gives a non-positive slope and drops the form (the reliability analogue of the
    # gaussian's downward-parabola check), while the monotone-agnostic forms still fit.
    forms = [r.form for r in fit_all([1, 2, 3, 4], [0.8, 0.6, 0.4, 0.2], Mode.FLOATING_POINT, 0)]
    assert "weibull" not in forms
    assert "quadratic" in forms


def test_logistic_recovers_growth_and_intercept_for_proportions_in_floating_point():
    # 44.2.17: y = 1/(1 + exp(-(0.8*x - 2))) — proportion data (every y < 1), so the fixed
    # ceiling a = 1 matches the true carrying capacity and the logit ln(y/(1 - y)) = b*x + c is
    # a line, recovering the growth rate b≈0.8 and intercept c≈-2 with error ≈ 0.
    xs = [0, 1, 2, 3, 4, 5]
    ys = [1 / (1 + math.exp(-(0.8 * x - 2))) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "logistic")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(1.0)  # proportions -> ceiling 1
    assert params["b"].to_float() == pytest.approx(0.8, rel=1e-6)
    assert params["c"].to_float() == pytest.approx(-2.0, rel=1e-6)
    assert result.error.to_float() == pytest.approx(0.0, abs=1e-9)
    assert "/(1 + exp(-(" in result.equation
    assert not result.error.exact


def test_logistic_ceiling_uses_headroom_above_the_max_for_non_proportion_data():
    # For non-proportion data (some y >= 1) the ceiling a is NOT fitted — it is the largest
    # observation lifted by 5% headroom, so no datum sits at the ceiling. Genuine sigmoid data
    # (saturating near 8) ranks the logistic among the best three, and its a = 1.05*max(y).
    xs = [0, 1, 2, 3, 4, 5, 6]
    ys = [8 / (1 + math.exp(-1.2 * (x - 3))) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "logistic")
    assert dict(result.parameters)["a"].to_float() == pytest.approx(1.05 * max(ys))


def test_logistic_is_dropped_in_rational_mode():
    # The logit needs ln/exp, which are irrational — rational mode refuses them, so the logistic
    # form drops while the polynomials remain (like the power/gaussian forms).
    ys = [1 / (1 + math.exp(-(0.8 * x - 2))) for x in (0, 1, 2, 3, 4, 5)]
    forms = [r.form for r in fit_all([0, 1, 2, 3, 4, 5], ys, Mode.RATIONAL, 0)]
    assert "logistic" not in forms
    assert "cubic" in forms


def test_logistic_is_dropped_when_some_y_is_not_positive():
    # The logit ln(y/(a - y)) needs y > 0; a non-positive y makes the ratio non-positive and the
    # log undefined, so the form drops cleanly while the log-free forms still fit.
    forms = [r.form for r in fit_all([0, 1, 2, 3], [0.2, 0.5, 0.8, 0.0], Mode.FLOATING_POINT, 0)]
    assert "logistic" not in forms
    assert "quadratic" in forms


def test_saturation_recovers_exactly_in_rational():
    # 44.2.11: y = x/(x + 2) (so a=1, b=2) — the double-reciprocal line 1/y = a + b*(1/x) is
    # exact in rational mode (reciprocals of rationals are rational), so a and b are recovered
    # on the nose with error 0. The intercept is a, the slope is b.
    result = _pick(fit_all([2, 0.5, 8], [0.5, 0.2, 0.8], Mode.RATIONAL, 0), "saturation")
    params = dict(result.parameters)
    assert params["a"].to_string() == "1"
    assert params["b"].to_string() == "2"
    assert result.equation == "x/(1*x + 2)"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_saturation_is_dropped_when_some_x_is_zero():
    # 1/x at x = 0 raises ZeroDivisionError; the fitter catches it as a clean DROP (no crash),
    # while the log-free / x=0-tolerant forms still fit and come back.
    forms = [r.form for r in fit_all([0, 1, 2, 3], [1, 2, 3, 4], Mode.RATIONAL, 0)]
    assert "saturation" not in forms
    assert "linear" in forms and "quadratic" in forms


def test_hyperbolic_recovers_exactly_in_rational():
    # 44.2.12: y = 1/(x + 1) (so a=1, b=1) — the line 1/y = a*x + b is exact in rational mode,
    # so a and b are recovered on the nose with error 0. The slope is a, the intercept is b.
    result = _pick(fit_all([1, 3, 4, 9], [0.5, 0.25, 0.2, 0.1], Mode.RATIONAL, 0), "hyperbolic")
    params = dict(result.parameters)
    assert params["a"].to_string() == "1"
    assert params["b"].to_string() == "1"
    assert result.equation == "1/(1*x + 1)"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_hyperbolic_is_dropped_when_some_y_is_zero():
    # 1/y at y = 0 raises ZeroDivisionError; the fitter catches it as a clean DROP (no crash),
    # while the forms that need no 1/y still fit and come back.
    forms = [r.form for r in fit_all([1, 2, 3, 4], [1, 0, 3, 4], Mode.RATIONAL, 0)]
    assert "hyperbolic" not in forms
    assert "quadratic" in forms  # forms needing no 1/y still fit (best-3 may trail off linear)


def test_generalized_hyperbolic_recovers_exactly_in_rational():
    # 44.2.18: y = 1/(x**2 + 1) (so a=1, b=0, c=1) — the reciprocal-quadratic 1/y = a*x**2 + b*x
    # + c is exact in rational mode (reciprocals of rationals are rational, the quadratic is
    # polynomial), so the coefficients are recovered on the nose with error 0.
    results = fit_all([0, 1, 2, 3], [1, 0.5, 0.2, 0.1], Mode.RATIONAL, 0)
    result = _pick(results, "generalized-hyperbolic")
    params = dict(result.parameters)
    assert params["a"].to_string() == "1"
    assert params["b"].to_string() == "0"
    assert params["c"].to_string() == "1"
    assert result.equation == "1/(1*x**2 + 0*x + 1)"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_generalized_hyperbolic_recovers_a_true_reciprocal_quadratic_in_floating_point():
    # y = 1/(2*x**2 + 3*x + 5) over six points — y is NOT a low-degree polynomial, so only the
    # reciprocal-quadratic fits it exactly, recovering a≈2, b≈3, c≈5 (the polynomials trail off).
    xs = [-2, -1, 0, 1, 2, 3]
    ys = [1 / (2 * x**2 + 3 * x + 5) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "generalized-hyperbolic")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(2.0, rel=1e-6)
    assert params["b"].to_float() == pytest.approx(3.0, rel=1e-6)
    assert params["c"].to_float() == pytest.approx(5.0, rel=1e-6)
    assert result.error.to_float() == pytest.approx(0.0, abs=1e-9)


def test_generalized_hyperbolic_is_dropped_when_some_y_is_zero():
    # 1/y at y = 0 raises ZeroDivisionError; the reciprocal-quadratic fitter catches it as a
    # clean DROP (no crash), while the forms that need no 1/y still fit and come back.
    forms = [r.form for r in fit_all([0, 1, 2, 3], [1, 0, 0.2, 0.1], Mode.FLOATING_POINT, 0)]
    assert "generalized-hyperbolic" not in forms
    assert "quadratic" in forms


def test_lorentzian_recovers_peak_centre_and_width_in_floating_point():
    # 44.2.19: y = 5/(1 + ((x-3)/2)**2) — the reciprocal-quadratic 1/y = A*x**2 + B*x + C is
    # fitted and the peak/centre/width recovered: peak a≈5, centre b≈3, half-width c≈2, error ≈ 0.
    xs = [0, 1, 2, 3, 4, 5, 6]
    ys = [5 / (1 + ((x - 3) / 2) ** 2) for x in xs]
    result = _pick(fit_all(xs, ys, Mode.FLOATING_POINT, 0), "lorentzian")
    params = dict(result.parameters)
    assert params["a"].to_float() == pytest.approx(5.0, rel=1e-6)  # the peak
    assert params["b"].to_float() == pytest.approx(3.0, rel=1e-6)  # the centre
    assert params["c"].to_float() == pytest.approx(2.0, rel=1e-6)  # the half-width
    assert result.error.to_float() == pytest.approx(0.0, abs=1e-9)
    assert "/(1 + ((x - " in result.equation and ")**2)" in result.equation
    assert not result.error.exact


def test_lorentzian_is_dropped_in_rational_mode():
    # The half-width recovery takes a sqrt, which is irrational — rational mode refuses it, so the
    # Lorentzian drops (like the gaussian), even though its reciprocal-quadratic sibling does not.
    ys = [5 / (1 + ((x - 3) / 2) ** 2) for x in (0, 1, 2, 3, 4, 5, 6)]
    forms = [r.form for r in fit_all([0, 1, 2, 3, 4, 5, 6], ys, Mode.RATIONAL, 0)]
    assert "lorentzian" not in forms
    assert "generalized-hyperbolic" in forms  # the sqrt-free sibling stays exact in rational


def test_lorentzian_is_dropped_when_data_is_not_peak_shaped():
    # A Lorentzian peaks, so its reciprocal 1/y is an UPWARD parabola; valley data (y = x**2 + 1)
    # makes 1/y a downward parabola (A < 0) and drops the form — the gaussian analogue of the
    # downward-parabola check — while the polynomial forms still fit.
    forms = [r.form for r in fit_all([-2, -1, 0, 1, 2], [5, 2, 1, 2, 5], Mode.FLOATING_POINT, 0)]
    assert "lorentzian" not in forms
    assert "quadratic" in forms


def test_lorentzian_is_dropped_when_some_y_is_zero():
    # 1/y at y = 0 raises ZeroDivisionError in the reciprocal-quadratic core; the fitter catches
    # it as a clean DROP (no crash), while the forms that need no 1/y still fit.
    forms = [r.form for r in fit_all([0, 1, 2, 3], [1, 0, 0.2, 0.1], Mode.FLOATING_POINT, 0)]
    assert "lorentzian" not in forms
    assert "quadratic" in forms


def test_laurent_recovers_exactly_in_rational():
    # 44.2.13: y = 2 + 3*x + 1/x (so a=2, b=3, c=1) — the direct basis {1, x, 1/x} is exact in
    # rational mode (a constant, x, and a reciprocal of a rational are all exact), so the three
    # coefficients are recovered on the nose with error 0. Three distinct x exactly determine them.
    result = _pick(fit_all([1, 2, 4], [6, 8.5, 14.25], Mode.RATIONAL, 0), "laurent")
    params = dict(result.parameters)
    assert params["a"].to_string() == "2"
    assert params["b"].to_string() == "3"
    assert params["c"].to_string() == "1"
    assert result.equation == "2 + 3*x + 1/x"
    assert result.error.to_string() == "0"
    assert result.error.exact


def test_laurent_is_dropped_when_some_x_is_zero():
    # The Laurent basis has a 1/x term, so an x = 0 datum raises ZeroDivisionError; the basis
    # fitter catches it as a clean DROP (no crash), while the forms with no 1/x still fit.
    forms = [r.form for r in fit_all([0, 1, 2, 3], [1, 2, 3, 4], Mode.RATIONAL, 0)]
    assert "laurent" not in forms
    assert "linear" in forms and "quadratic" in forms


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
    # 44.5: three points with a zero x AND a zero y in rational mode drops almost everything —
    # the cubic is underdetermined, the reciprocal and saturation hit 1/0 at x = 0, the
    # hyperbolic and saturation hit 1/0 at y = 0, and the power/log/sqrt forms need x > 0 or
    # refuse their irrational basis — so only the linear and quadratic forms fit and fewer than
    # three come back, not padded.
    results = fit_all([0, 1, 2], [2, 5, 0], Mode.RATIONAL, 0)
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
    # 44.2.1-44.2.19: the registry declares linear plus the quadratic/cubic/power, the
    # exponential/logarithmic/square-root/reciprocal forms, the sinusoid, the gaussian, the
    # saturation/hyperbolic reciprocal-line forms, the Laurent direct-basis form, the
    # exp-reciprocal / Arrhenius law, the Hoerl model, the Weibull CDF, the fixed-L logistic, the
    # generalized hyperbolic, and the Lorentzian peak — in TODO numeric order — each with its
    # parameter names, model template and domain limit.
    by_name = {form.name: form for form in CURVE_FORMS}
    assert list(by_name) == [
        "linear",
        "quadratic",
        "cubic",
        "power",
        "exponential",
        "logarithmic",
        "square-root",
        "reciprocal",
        "sinusoidal",
        "gaussian",
        "saturation",
        "hyperbolic",
        "laurent",
        "exp-reciprocal",
        "hoerl",
        "weibull",
        "logistic",
        "generalized-hyperbolic",
        "lorentzian",
    ]
    assert by_name["linear"].parameters == ("a", "b")
    assert by_name["quadratic"].parameters == ("a", "b", "c")
    assert by_name["quadratic"].template == "a*x**2 + b*x + c"
    assert by_name["cubic"].parameters == ("a", "b", "c", "d")
    assert by_name["cubic"].template == "a*x**3 + b*x**2 + c*x + d"
    # Only the power form restricts x; the polynomial forms accept any x.
    assert by_name["power"].domain == "x > 0"
    assert by_name["linear"].domain is None
    # The four new closed-form forms, each two-parameter, with their templates and domains.
    assert by_name["exponential"].parameters == ("a", "b")
    assert by_name["exponential"].template == "a*exp(b*x)"
    assert by_name["exponential"].domain == "y > 0"  # restricts y, not x
    assert by_name["logarithmic"].template == "a + b*ln(x)"
    assert by_name["logarithmic"].domain == "x > 0"
    assert by_name["square-root"].template == "a*sqrt(x) + b"
    assert by_name["square-root"].domain == "x >= 0"
    assert by_name["reciprocal"].template == "a/x + b"
    assert by_name["reciprocal"].domain == "x != 0"
    # The sinusoid: four parameters, and domain None — the rational limitation is a mode
    # drop (irrational sin), not an x restriction.
    assert by_name["sinusoidal"].parameters == ("a", "b", "c", "d")
    assert by_name["sinusoidal"].template == "a*sin(b*x + c) + d"
    assert by_name["sinusoidal"].domain is None
    # The gaussian: three parameters (peak/centre/width), and domain y > 0 (its logs need
    # positive y), fitted in closed form via Caruana's method.
    assert by_name["gaussian"].parameters == ("a", "b", "c")
    assert by_name["gaussian"].template == "a*exp(-(x-b)**2/(2*c**2))"
    assert by_name["gaussian"].domain == "y > 0"
    # The final two forms: the saturation (Michaelis-Menten) and hyperbolic curves, each
    # two-parameter, fitted in closed form via a reciprocal-line transform.
    assert by_name["saturation"].parameters == ("a", "b")
    assert by_name["saturation"].template == "x/(a*x + b)"
    assert by_name["saturation"].domain == "x != 0, y != 0"
    assert by_name["hyperbolic"].parameters == ("a", "b")
    assert by_name["hyperbolic"].template == "1/(a*x + b)"
    assert by_name["hyperbolic"].domain == "y != 0"
    # The Laurent form (44.2.13): three parameters over the direct basis {1, x, 1/x}, domain
    # x != 0 (its 1/x pole), fitted in closed form with no transform.
    assert by_name["laurent"].parameters == ("a", "b", "c")
    assert by_name["laurent"].template == "a + b*x + c/x"
    assert by_name["laurent"].domain == "x != 0"
    # The exp-reciprocal / Arrhenius form (44.2.14): two parameters, log-linearised against 1/x,
    # domain x != 0 (the reciprocal) and y > 0 (the log).
    assert by_name["exp-reciprocal"].parameters == ("a", "b")
    assert by_name["exp-reciprocal"].template == "a*exp(b/x)"
    assert by_name["exp-reciprocal"].domain == "x != 0, y > 0"
    # The Hoerl model (44.2.15): three parameters, log-linearised over the basis {1, x, ln x},
    # domain x > 0 and y > 0 (both logs).
    assert by_name["hoerl"].parameters == ("a", "b", "c")
    assert by_name["hoerl"].template == "a*b**x*x**c"
    assert by_name["hoerl"].domain == "x > 0, y > 0"
    # The Weibull CDF (44.2.16): two parameters (scale a, shape b), fitted by the double-log
    # Weibull plot, domain x > 0 and 0 < y < 1 (the two logs).
    assert by_name["weibull"].parameters == ("a", "b")
    assert by_name["weibull"].template == "1 - exp(-(x/a)**b)"
    assert by_name["weibull"].domain == "x > 0, 0 < y < 1"
    # The logistic (44.2.17): three parameters (ceiling a — data-fixed, not fitted — growth rate
    # b, intercept c), fitted by the logit once a is fixed, domain 0 < y < a.
    assert by_name["logistic"].parameters == ("a", "b", "c")
    assert by_name["logistic"].template == "a/(1 + exp(-(b*x + c)))"
    assert by_name["logistic"].domain == "0 < y < a"
    # The generalized hyperbolic (44.2.18): three parameters over a reciprocal-quadratic
    # (1/y = a*x**2 + b*x + c, coefficients read directly), domain y != 0; exact in rational.
    assert by_name["generalized-hyperbolic"].parameters == ("a", "b", "c")
    assert by_name["generalized-hyperbolic"].template == "1/(a*x**2 + b*x + c)"
    assert by_name["generalized-hyperbolic"].domain == "y != 0"
    # The Lorentzian / Cauchy peak (44.2.19): three parameters (peak a, centre b, half-width c),
    # the same reciprocal-quadratic with peak/centre/width recovered, domain y != 0.
    assert by_name["lorentzian"].parameters == ("a", "b", "c")
    assert by_name["lorentzian"].template == "a/(1 + ((x-b)/c)**2)"
    assert by_name["lorentzian"].domain == "y != 0"


def test_every_form_is_now_wired_to_a_fitter():
    # 44.3: all four declared forms carry a live closed-form fitter.
    assert all(form.fit is not None for form in CURVE_FORMS)


def test_linear_basis_fitter_fits_a_non_polynomial_basis_exactly():
    # 44.9: the shared fitter is over an ARBITRARY basis, not just powers of x. Over the
    # reciprocal basis {1, 1/x} the model is c0 + c1/x — linear in its parameters though 1/x
    # is curved — so y = 3 + 2/x is recovered exactly in rational mode (error 0, on the nose).
    one = lambda x, mode, scale: Value.from_real(1, mode, scale)  # noqa: E731
    recip = lambda x, mode, scale: Value.from_real(1, mode, scale).div(x)  # noqa: E731
    render = lambda coeffs: f"{coeffs[0].to_string()} + {coeffs[1].to_string()}/x"  # noqa: E731
    fit = _linear_basis_fitter("reciprocal", ("c0", "c1"), (one, recip), render)
    xs = [Value.from_real(v, Mode.RATIONAL, 12) for v in (1, 2, 4)]
    ys = [Value.from_real(v, Mode.RATIONAL, 12) for v in (5, 4, 3.5)]  # 3 + 2/x
    result = fit(xs, ys, Mode.RATIONAL, 12)
    params = dict(result.parameters)
    assert params["c0"].to_string() == "3"
    assert params["c1"].to_string() == "2"
    assert result.equation == "3 + 2/x"
    assert result.error.to_string() == "0"
    assert result.error.exact
