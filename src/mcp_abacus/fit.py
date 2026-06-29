# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The fit tool's engine: try each known curve form on paired x/y data (TODO 44).

The caller hands over observations ``(x_i, y_i)`` and the engine estimates each curve
form's free parameters, reporting the fitted equation and a residual error. Like the
solver it is type-faithful: every datum is materialised as a mode-faithful ``Value`` and
the whole fit is computed in the active mode — exact in rational, mode-rounded in
fixed-point / floating-point — so the precision verdict carries through to the answer.

Each form lives in ``CURVE_FORMS`` (44.2), the registry the tool iterates; a form
declares its name, model template, parameter names and any domain limit, and carries the
fitter that estimates its parameters. Every form wired today is fitted in CLOSED FORM
(44.3) rather than through the solver's iterative optimise engine, because each has an
exact least-squares solution:

  - linear / quadratic / cubic are LINEAR IN THEIR PARAMETERS, so their least-squares fit
    is the solution of the normal equations ``(Vᵀ·V)·c = Vᵀ·y`` (V the Vandermonde matrix
    of the data) — solved by Gauss-Jordan elimination in mode-faithful Value arithmetic,
    so the coefficients are EXACT in rational mode, mode-rounded in fixed-point, and a
    native double in floating-point (``_fit_linear`` / ``_fit_polynomial``).
  - power ``a·x**b`` is intrinsically non-linear but LINEARISES under logs —
    ``ln y = ln a + b·ln x`` is a straight line in ``(ln x, ln y)`` — so it is the same
    line fit on the logged data, with ``b`` the slope and ``a = exp(intercept)``
    (``_fit_power``). The logs make it inexact wherever the mode rounds them; in rational
    mode they are irrational and the type refuses them, so the power form is simply
    dropped there (its honest exact-or-refuse outcome).

The genuinely non-linearisable forms still to come (the sinusoid, 44.2.9) are what will
need the solver's optimise engine; this module does not reach for it yet. ``fit_all``
fits every form, DROPPING any that cannot fit the data (a domain miss, a degenerate
configuration, or a mode that cannot represent the fit) rather than failing the whole
request (44.3); the cross-form error ranking and best-3 selection (44.4-44.5) are still to
come, so it returns every fitted form unranked.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mcp_abacus.expr.value import Mode, NotRepresentableError, Value

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

    ``name`` is the form's name and ``parameters`` its free-parameter names in equation
    order. ``template`` is the model written over the variable ``x`` with those parameters
    (e.g. ``"a*x**2 + b*x + c"``) — the named template 44.2 calls for, which a fitter
    substitutes fitted values into and the 44.3 optimiser will evaluate. ``domain`` records
    any restriction the form places on ``x`` (e.g. ``"x > 0"`` for the power form), or
    ``None`` when ``x`` is unrestricted. ``fit`` is the fitter: it takes the data as
    mode-faithful Values plus the mode and working scale and returns a FitResult (or raises
    FitError when the data cannot support this form, e.g. a vertical line for the linear
    fit). It is ``None`` for a form that is declared but not yet wired to a fitter — the
    quadratic, cubic and power forms (44.2.2-44.2.4) are declared today but estimated only
    once the solver-driven fit (44.3) lands, and ``fit_all`` skips a form whose ``fit`` is
    ``None``. The registry ``CURVE_FORMS`` holds one entry per form; the tool iterates it.
    """

    name: str
    parameters: tuple[str, ...]
    template: str
    domain: "str | None"
    fit: "Callable[[list[Value], list[Value], Mode, int], FitResult] | None"


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


def _term_body(magnitude: Value, power: int) -> str:
    """Render one unsigned polynomial term ``magnitude·x**power`` for the equation string.

    ``power`` 0 is the bare constant (``magnitude``), 1 is ``magnitude*x``, and higher
    powers are ``magnitude*x**power``. ``magnitude`` is rendered verbatim, so a *leading*
    term passes its coefficient straight in (its sign included) while a non-leading term
    passes the magnitude with the sign already split off (see ``_polynomial_equation``).
    """
    body = magnitude.to_string()
    if power == 0:
        return body
    if power == 1:
        return f"{body}*x"
    return f"{body}*x**{power}"


def _polynomial_equation(coeffs: list[Value]) -> str:
    """Render coefficients (highest degree first) as ``a*x**2 + b*x + c`` over ``x``.

    The leading term keeps its own sign (``-2*x**2``); each later term reads its sign off
    the coefficient and renders ``+ body`` or ``- |body|`` so the equation flows naturally
    (``2.5*x - 1`` rather than ``2.5*x + -1``). The result is valid calculate source over
    ``x`` (44.5), so it pastes straight back in. Every term is kept — a zero coefficient
    still prints (``0*x**2 + …``) so the form is visible in the rendered equation.
    """
    degree = len(coeffs) - 1
    parts = [_term_body(coeffs[0], degree)]  # leading term carries its own sign
    for i, coeff in enumerate(coeffs[1:], start=1):
        negative = coeff.to_float() < 0
        magnitude = coeff.neg() if negative else coeff
        parts.append(f"{'-' if negative else '+'} {_term_body(magnitude, degree - i)}")
    return " ".join(parts)


def _line_coeffs(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> tuple[Value, Value]:
    """Least-squares ``(slope, intercept)`` of the line through the data (44.2.1).

    Solves the line's normal equations in closed form — ``slope = (n·Σxy − Σx·Σy) /
    (n·Σx² − (Σx)²)`` and ``intercept = (Σy − slope·Σx) / n`` — entirely through
    mode-faithful Value arithmetic, so both behave exactly as the active type would (exact
    in rational, rounded at the scale in fixed-point, native double in floating-point).
    The linear fit uses it directly; the power fit (44.2.4) uses it on the logged data.

    Raises FitError when every ``x`` is equal (the slope is undefined). That degeneracy is
    detected DIRECTLY — all the ``x`` sharing one value — rather than by testing the
    denominator ``n·Σx² − (Σx)²`` for zero: the two are mathematically equivalent (the
    denominator is ``n²·Var(x)``), but in fixed-point the denominator can round to a tiny
    NON-zero value for identical inputs (``9·round(u²) ≠ round(9·u²)``), which would slip a
    garbage near-vertical fit through — most visibly the power form, whose ``ln x`` are all
    equal. The caller guarantees at least two points.
    """
    if all(x.to_string() == xs[0].to_string() for x in xs):
        raise FitError(
            "Cannot fit: every x value is equal, so the slope is undefined. "
            "Provide data with at least two distinct x values."
        )
    n = len(xs)
    big_n = Value.from_real(n, mode, scale)
    sum_x = _sum(xs, mode, scale)
    sum_y = _sum(ys, mode, scale)
    sum_xx = _sum([x.mul(x) for x in xs], mode, scale)
    sum_xy = _sum([x.mul(y) for x, y in zip(xs, ys, strict=True)], mode, scale)

    denominator = big_n.mul(sum_xx).sub(sum_x.mul(sum_x))
    if denominator.to_float() == 0.0:
        raise FitError(
            "Cannot fit: the x values are too close to determine a slope. "
            "Provide data with at least two distinct x values."
        )
    slope = big_n.mul(sum_xy).sub(sum_x.mul(sum_y)).div(denominator)
    intercept = sum_y.sub(slope.mul(sum_x)).div(big_n)
    return slope, intercept


def _sum_squared_residuals(
    model: list[Value], ys: list[Value], mode: Mode, scale: int
) -> Value:
    """The fit error ``Σ (modelᵢ − yᵢ)²`` in the active mode (44.4 preview)."""
    residuals = [m.sub(y) for m, y in zip(model, ys, strict=True)]
    return _sum([r.mul(r) for r in residuals], mode, scale)


def _fit_linear(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Least-squares fit of the line ``a*x + b`` to the data (44.2.1).

    A thin wrapper over :func:`_line_coeffs`: the slope is ``a`` and the intercept ``b``,
    both in mode-faithful Value arithmetic, and the error is the sum of squared residuals
    ``Σ (a·xᵢ + b − yᵢ)²`` in that same type. Raises FitError on a vertical line (every
    ``x`` equal); the caller guarantees at least two points.
    """
    a, b = _line_coeffs(xs, ys, mode, scale)
    model = [a.mul(x).add(b) for x in xs]
    error = _sum_squared_residuals(model, ys, mode, scale)
    equation = _polynomial_equation([a, b])
    return FitResult("linear", equation, (("a", a), ("b", b)), error)


def _solve_linear_system(
    matrix: list[list[Value]], rhs: list[Value], name: str, mode: Mode, scale: int
) -> list[Value]:
    """Solve the square system ``matrix·c = rhs`` for ``c`` by Gauss-Jordan elimination.

    The elimination runs entirely in mode-faithful Value arithmetic — exact in rational,
    mode-rounded in fixed-point, native double in floating-point — so the solution carries
    the active type's own precision. Partial pivoting (largest-magnitude pivot, chosen on
    the float shadow purely for numerical stability) guards the elimination; a pivot that
    is zero means the normal-equations matrix is singular — the data does not determine the
    ``name`` form's coefficients (too few distinct ``x``) — which is a FitError so the form
    is dropped, not fatal (44.3).
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]  # augmented [matrix | rhs]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col].to_float()))
        if aug[pivot][col].to_float() == 0.0:
            raise FitError(
                f"Cannot fit {name}: the data does not determine its {n} coefficients "
                f"(it needs at least {n} distinct x values)."
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col].div(pivot_val)
            aug[r] = [aug[r][c].sub(factor.mul(aug[col][c])) for c in range(n + 1)]
    return [aug[i][n].div(aug[i][i]) for i in range(n)]


def _polynomial_fitter(
    name: str, degree: int, parameters: tuple[str, ...]
) -> "Callable[[list[Value], list[Value], Mode, int], FitResult]":
    """Build the fitter for the degree-``degree`` polynomial form (44.2.2 / 44.2.3).

    A polynomial is linear in its coefficients, so its least-squares fit is the solution of
    the normal equations ``(Vᵀ·V)·c = Vᵀ·y``: the matrix is the symmetric Hankel of power
    sums ``S[j+k] = Σ xᵢ**(j+k)`` and the right-hand side is ``T[j] = Σ xᵢ**j·yᵢ``, both
    accumulated in mode-faithful Value arithmetic so the whole solve stays exact in rational
    and mode-rounded otherwise. ``parameters`` names the coefficients highest-degree first
    (``a`` the leading coefficient), matching the rendered equation. The returned fitter has
    the CurveForm.fit signature and raises FitError when the data is degenerate (fewer than
    ``degree + 1`` distinct ``x`` — a singular matrix).
    """

    def fit(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
        # One pass over the data: per point build xᵢ**0 … xᵢ**(2·degree) by repeated
        # multiplication (keeping the mode's own rounding), retaining the low half
        # (xᵢ**0 … xᵢ**degree) to rebuild the model after the solve. From these come the
        # power sums S[0..2·degree] and the right-hand side T[0..degree] of the normal eqs.
        one = Value.from_real(1, mode, scale)
        point_powers: list[list[Value]] = []
        s_terms: list[list[Value]] = [[] for _ in range(2 * degree + 1)]
        t_terms: list[list[Value]] = [[] for _ in range(degree + 1)]
        for x, y in zip(xs, ys, strict=True):
            powers = [one]
            for _ in range(2 * degree):
                powers.append(powers[-1].mul(x))
            point_powers.append(powers[: degree + 1])
            for m in range(2 * degree + 1):
                s_terms[m].append(powers[m])
            for j in range(degree + 1):
                t_terms[j].append(powers[j].mul(y))
        power_sums = [_sum(terms, mode, scale) for terms in s_terms]
        rhs = [_sum(terms, mode, scale) for terms in t_terms]
        matrix = [[power_sums[j + k] for k in range(degree + 1)] for j in range(degree + 1)]
        # Solve for c[0..degree] (constant-first), then read it highest-degree-first.
        low_to_high = _solve_linear_system(matrix, rhs, name, mode, scale)
        coeffs = list(reversed(low_to_high))  # a (leading) … constant
        model = [
            _sum([low_to_high[k].mul(p) for k, p in enumerate(powers)], mode, scale)
            for powers in point_powers
        ]
        error = _sum_squared_residuals(model, ys, mode, scale)
        equation = _polynomial_equation(coeffs)
        return FitResult(name, equation, tuple(zip(parameters, coeffs, strict=True)), error)

    return fit


def _fit_power(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Least-squares fit of the power law ``a*x**b`` to the data (44.2.4); domain ``x > 0``.

    The power law linearises under logs — ``ln y = ln a + b·ln x`` is a straight line in
    ``(ln x, ln y)`` — so the fit is :func:`_line_coeffs` on the logged data: ``b`` is the
    slope and ``a = exp(intercept)``, every step in mode-faithful Value arithmetic. The
    error is the sum of squared residuals ``Σ (a·xᵢ**b − yᵢ)²`` of the ORIGINAL model in the
    active mode, so it measures the fit where the data lives, not in log space.

    The model is evaluated as ``a·exp(b·ln xᵢ)`` rather than ``a·(xᵢ**b)``: they are equal
    for ``xᵢ > 0``, but ``**`` with a non-integer exponent first tries an EXACT root, which
    for the fit's noisy ``b`` (e.g. ``1.500000012``) means raising ``xᵢ`` to a
    hundred-million-scale numerator — astronomically slow — whereas ``exp(b·ln x)`` is the
    direct, fast series. The logs are reused from the linearisation.

    Raises FitError (so the form is dropped, 44.3) when the logs are undefined or
    unrepresentable: any ``xᵢ ≤ 0`` or ``yᵢ ≤ 0`` has no real log, and in rational mode the
    logs and ``exp`` are irrational — the type refuses them — so the power form simply does
    not fit there. NotRepresentableError from any of those is converted to FitError.
    """
    try:
        log_x = [x.ln() for x in xs]
        log_y = [y.ln() for y in ys]
        slope, intercept = _line_coeffs(log_x, log_y, mode, scale)
        b = slope
        a = intercept.exp()
        model = [a.mul(b.mul(lx).exp()) for lx in log_x]  # a·exp(b·ln x) == a·x**b for x>0
        error = _sum_squared_residuals(model, ys, mode, scale)
    except NotRepresentableError as exc:
        raise FitError(
            "Cannot fit a*x**b: it needs x > 0 and y > 0, and a mode that can represent "
            f"their logarithms (rational refuses the irrational logs). {exc}"
        ) from exc
    b_str = f"({b.to_string()})" if b.to_float() < 0 else b.to_string()
    equation = f"{a.to_string()}*x**{b_str}"
    return FitResult("power", equation, (("a", a), ("b", b)), error)


# The curve library (44.2): one entry per model form, iterated by the fit tool. Each entry
# DECLARES the form — its name, model template over x, free-parameter names and any domain
# limit — and carries the closed-form fitter that estimates its parameters (44.3). Linear,
# quadratic and cubic share the polynomial normal-equations solver (linear via the line's
# direct formula); the power form linearises under logs. The exponential, logarithmic, …
# forms (44.2.5+) slot in the same way as they land, each with its own fitter.
LINEAR = CurveForm("linear", ("a", "b"), "a*x + b", None, _fit_linear)
QUADRATIC = CurveForm(
    "quadratic", ("a", "b", "c"), "a*x**2 + b*x + c", None,
    _polynomial_fitter("quadratic", 2, ("a", "b", "c")),
)
CUBIC = CurveForm(
    "cubic", ("a", "b", "c", "d"), "a*x**3 + b*x**2 + c*x + d", None,
    _polynomial_fitter("cubic", 3, ("a", "b", "c", "d")),
)
POWER = CurveForm("power", ("a", "b"), "a*x**b", "x > 0", _fit_power)
CURVE_FORMS: tuple[CurveForm, ...] = (LINEAR, QUADRATIC, CUBIC, POWER)


def fit_all(xs: Sequence[float], ys: Sequence[float], mode: Mode, floor: int) -> list[FitResult]:
    """Fit every curve form in the library to the data, returning each form's FitResult (44.1).

    The data are materialised once as mode-faithful Values at the working scale, then each
    form in ``CURVE_FORMS`` is fitted against them. A form that cannot fit *this* data — a
    domain miss (power needs ``x > 0``, ``y > 0``), a degenerate configuration (too few
    distinct ``x`` for a polynomial's order), or a mode that cannot represent the fit (the
    power form's irrational logs in rational mode) — is DROPPED rather than fatal (44.3
    drop-and-continue), so the others still come back.

    The survivors are RANKED by their fit error — the sum of squared residuals (44.4) — so
    the best (least-error) fit comes first (44.5). The error is the comparable goodness
    number across forms; ties (e.g. several forms fitting the data exactly, error 0) keep
    registry order, which is ascending in complexity, so the simplest exact form leads. The
    best-N truncation is a later refinement; every fitted form is returned for now.

    If no form fits at all, the dropping would otherwise swallow the reason, so the FIRST
    form's FitError is re-raised — the linear form's, the most fundamental (e.g. its
    vertical-line refusal when every ``x`` is equal).
    """
    scale = _fit_scale(mode, floor)
    x_values = [Value.from_real(x, mode, scale) for x in xs]
    y_values = [Value.from_real(y, mode, scale) for y in ys]
    results: list[FitResult] = []
    first_error: FitError | None = None
    for form in CURVE_FORMS:
        if form.fit is None:
            continue  # declared but not yet wired to a fitter (a later form)
        try:
            results.append(form.fit(x_values, y_values, mode, scale))
        except FitError as exc:
            if first_error is None:
                first_error = exc  # remember the first reason; keep trying the rest
    if not results and first_error is not None:
        raise first_error  # nothing fit — surface the first (linear) reason
    # Rank best (least error) first; a stable sort keeps registry order on ties (44.5).
    results.sort(key=lambda result: result.error.to_float())
    return results
