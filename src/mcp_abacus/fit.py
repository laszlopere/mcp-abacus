# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The fit tool's engine: try each known curve form on paired x/y data (TODO 44).

The caller hands over observations ``(x_i, y_i)`` and the engine estimates each curve
form's free parameters, reporting the fitted equation and a residual error. Like the
solver it is type-faithful: every datum is materialised as a mode-faithful ``Value`` and
the whole fit is computed in the active mode — exact in rational, mode-rounded in
fixed-point / floating-point — so the precision verdict carries through to the answer.

Each form lives in ``CURVE_FORMS`` (44.2), the registry the tool iterates; a form knows
its parameter names and how to fit itself. Only the LINEAR form ``a*x + b`` exists today
(44.2.1): a straight line has a closed-form least-squares solution (the normal
equations), so it is fitted directly here rather than through the solver's iterative
optimise engine — that general machinery (44.3), which the non-linear forms will need,
is a separate item. The cross-form parameter search, error ranking, and best-3 selection
(44.3-44.5) are likewise still to come; for now ``fit_all`` returns the one linear fit.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mcp_abacus.expr.value import Mode, Value

_RATIONAL_FIT_DECIMALS = 12  # rational has no scale of its own; materialise data at this one


class FitError(Exception):
    """A fit request that cannot be honoured — bad data or a form that cannot fit it.

    Like SolverError it carries no source line: a fit failure is about the data (too few
    points, mismatched lengths, a degenerate configuration) or the outcome, not a
    position in any expression text.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class FitResult:
    """One curve form fitted to the data — the outcome the tool renders (44.1 / 44.5).

    ``form`` names the curve (e.g. ``"linear"``); ``equation`` is the fitted model with
    its parameters substituted, written in the calculate language over the variable ``x``
    (so it can be pasted straight into ``calculate`` with ``x`` bound). ``parameters`` is
    the ordered ``(name, Value)`` for each free parameter, and ``error`` is the residual
    error of the fit (the sum of squared residuals) — every Value mode-faithful, so its
    own exact/precision verdict describes how exact the fit is. The server turns this into
    the reply dict; a bad request is a FitError instead, never a FitResult.
    """

    form: str
    equation: str
    parameters: tuple[tuple[str, Value], ...]
    error: Value


@dataclass(frozen=True, slots=True)
class CurveForm:
    """A candidate model form in the curve library (44.2).

    ``name`` is the form's name, ``parameters`` its free-parameter names in equation
    order, and ``fit`` the fitter: it takes the data as mode-faithful Values plus the
    mode and working scale, and returns a FitResult (or raises FitError when the data
    cannot support this form, e.g. a vertical line for the linear fit). The registry
    ``CURVE_FORMS`` holds one entry per form; the tool iterates it.
    """

    name: str
    parameters: tuple[str, ...]
    fit: "Callable[[list[Value], list[Value], Mode, int], FitResult]"


def _fit_scale(mode: Mode, floor: int) -> int:
    """The decimal scale data is materialised at, per mode (mirrors solver._search_scale).

    Fixed-point binds each datum at the run's floor (min_fixed_point_precision), so the
    fit's arithmetic keeps that many fractional digits rather than collapsing to whole
    numbers. Rational has no inherent scale, so it materialises at a fixed working
    precision. Floating-point ignores the scale (a double has none); 0 is an unused
    placeholder.
    """
    match mode:
        case Mode.FIXED_POINT:
            return floor
        case Mode.RATIONAL:
            return _RATIONAL_FIT_DECIMALS
        case Mode.FLOATING_POINT:
            return 0
        case _:
            raise FitError(f"the fit engine does not support {mode.value} mode")


def _sum(values: list[Value], mode: Mode, scale: int) -> Value:
    """Fold ``values`` with the mode's own ``+``, seeded at a mode-faithful zero."""
    total = Value.from_real(0, mode, scale)
    for v in values:
        total = total.add(v)
    return total


def _linear_term(coeff: Value, var: str) -> str:
    """Render ``coeff*var`` or ``+ coeff`` for the equation string, sign read off the value.

    A negative coefficient renders as ``- |coeff|`` so the equation reads naturally
    (``2.5*x - 1`` rather than ``2.5*x + -1``); the constant term passes ``var=""``.
    """
    negative = coeff.to_float() < 0
    magnitude = coeff.neg() if negative else coeff
    body = f"{magnitude.to_string()}*{var}" if var else magnitude.to_string()
    return f"- {body}" if negative else f"+ {body}"


def _fit_linear(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Least-squares fit of the line ``a*x + b`` to the data (44.2.1).

    Solves the normal equations in closed form — ``a = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)``
    and ``b = (Σy − a·Σx) / n`` — entirely through mode-faithful Value arithmetic, so the
    slope and intercept behave exactly as the active type would (exact in rational, rounded
    at the scale in fixed-point, native double in floating-point). The error is the sum of
    squared residuals ``Σ (a·xᵢ + b − yᵢ)²`` in that same type.

    Raises FitError when every ``x`` is equal (a vertical line — the denominator is zero,
    so the slope is undefined); the caller guarantees at least two points.
    """
    n = len(xs)
    big_n = Value.from_real(n, mode, scale)
    sum_x = _sum(xs, mode, scale)
    sum_y = _sum(ys, mode, scale)
    sum_xx = _sum([x.mul(x) for x in xs], mode, scale)
    sum_xy = _sum([x.mul(y) for x, y in zip(xs, ys, strict=True)], mode, scale)

    denominator = big_n.mul(sum_xx).sub(sum_x.mul(sum_x))
    if denominator.to_float() == 0.0:
        raise FitError(
            "Cannot fit a line: every x value is equal, so the slope is undefined. "
            "Provide data with at least two distinct x values."
        )
    a = big_n.mul(sum_xy).sub(sum_x.mul(sum_y)).div(denominator)
    b = sum_y.sub(a.mul(sum_x)).div(big_n)

    residuals = [a.mul(x).add(b).sub(y) for x, y in zip(xs, ys, strict=True)]
    error = _sum([r.mul(r) for r in residuals], mode, scale)
    equation = f"{a.to_string()}*x {_linear_term(b, '')}"
    return FitResult("linear", equation, (("a", a), ("b", b)), error)


# The curve library (44.2): one entry per model form, iterated by the fit tool. Only the
# linear form is wired today (44.2.1); the quadratic, power, exponential, … forms (44.2.2+)
# slot in here as they land, each a CurveForm with its own fitter.
LINEAR = CurveForm("linear", ("a", "b"), _fit_linear)
CURVE_FORMS: tuple[CurveForm, ...] = (LINEAR,)


def fit_all(xs: Sequence[float], ys: Sequence[float], mode: Mode, floor: int) -> list[FitResult]:
    """Fit every curve form in the library to the data, returning each form's FitResult (44.1).

    The data are materialised once as mode-faithful Values at the working scale, then each
    form in ``CURVE_FORMS`` is fitted against them. With only the linear form wired today
    (44.2.1) this returns a single result; the cross-form ranking and best-3 selection
    (44.5) arrive with the rest of the library. A form that cannot fit the data raises
    FitError, which propagates (the drop-and-continue behaviour is part of 44.3).
    """
    scale = _fit_scale(mode, floor)
    x_values = [Value.from_real(x, mode, scale) for x in xs]
    y_values = [Value.from_real(y, mode, scale) for y in ys]
    return [form.fit(x_values, y_values, mode, scale) for form in CURVE_FORMS]
