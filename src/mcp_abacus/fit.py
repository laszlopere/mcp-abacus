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
fitter that estimates its parameters. Every form is fitted in CLOSED FORM (44.3) EXCEPT
the sinusoid (44.2.9), which has no closed-form least-squares solution and is fitted by an
iterative frequency search instead. The closed-form forms each have an exact least-squares
solution:

  - linear / quadratic / cubic are LINEAR IN THEIR PARAMETERS, so their least-squares fit
    is the solution of the normal equations ``(Bᵀ·B)·c = Bᵀ·y`` (B the design matrix
    ``Bᵢⱼ = fⱼ(xᵢ)`` over the form's basis functions) — solved by Gauss-Jordan elimination
    in mode-faithful Value arithmetic, so the coefficients are EXACT in rational mode,
    mode-rounded in fixed-point, and a native double in floating-point. The quadratic and
    cubic share one general basis fitter (``_linear_basis_fitter``, 44.9) over the power
    basis ``{x**degree, …, x, 1}``; the logarithmic ``a + b·ln x`` (basis ``{1, ln x}``),
    square-root ``a·√x + b`` (basis ``{√x, 1}``), reciprocal ``a/x + b`` (basis ``{1/x, 1}``)
    and Laurent ``a + b·x + c/x`` (basis ``{1, x, 1/x}``, 44.2.13) forms are linear in their
    own curved basis, so each just wires its basis into that same fitter rather than
    re-deriving the normal equations. The reciprocal and Laurent are exact in every mode
    (``1/x`` of a rational is rational, and ``x`` / the constant are exact); the logarithmic
    and square-root are irrational in rational mode (like the power form), so they drop there.
    The linear form keeps the line's direct two-coefficient formula (``_fit_linear`` /
    ``_line_coeffs``),
    which the power and exponential fits reuse on logged data.
  - power ``a·x**b``, exponential ``a·exp(b·x)`` and exp-reciprocal ``a·exp(b/x)`` (44.2.14,
    the Arrhenius law) are intrinsically non-linear but LINEARISE under logs. Power logs both
    axes — ``ln y = ln a + b·ln x`` is a straight line in ``(ln x, ln y)`` (``_fit_power``);
    exponential logs only ``y`` — ``ln y = ln a + b·x`` is a line in ``(x, ln y)``
    (``_fit_exponential``); exp-reciprocal logs ``y`` against the RECIPROCAL of ``x`` — ``ln y
    = ln a + b·(1/x)`` is a line in ``(1/x, ln y)`` (``_fit_exp_reciprocal``) — and all recover
    ``b`` from the slope and ``a = exp(intercept)``. The logs make them inexact wherever the
    mode rounds them; in rational mode the logs are irrational and the type refuses them, so all
    three are simply dropped there (their honest exact-or-refuse outcome).
  - the gaussian ``a·exp(−(x−b)²/(2c²))`` (44.2.10) also linearises under logs, but to a
    QUADRATIC rather than a line — Caruana's method: ``ln y = p₂·x² + p₁·x + p₀``, so the fit
    least-squares-fits that quadratic over the power basis ``{x², x, 1}`` (the same normal
    equations) and RECOVERS the peak/centre/width from the coefficients (``c = √(−1/2p₂)``,
    ``b = −p₁/2p₂``, ``a = exp(p₀ − p₂·b²)``), needing a downward parabola (``p₂ < 0``);
    non-positive ``y`` has no log and rational refuses the irrational logs/exp/√, so it drops
    there too (``_fit_gaussian``).
  - the Hoerl model ``a·b**x·x**c`` (44.2.15, a power law modulated by an exponential) is the
    gaussian's structural twin: it logs to a LINE over TWO regressors — ``ln y = ln a +
    (ln b)·x + c·ln x`` — linear in its coefficients over the basis ``{1, x, ln x}``, so the
    fit least-squares-fits that basis to ``(x, ln y)`` by the SAME normal equations
    (``_solve_linear_system``) and RECOVERS ``a = exp(p₀)``, ``b = exp(p₁)``, ``c = p₂`` from the
    coefficients; the rebuilt model ``exp(p₀ + p₁·x + p₂·ln x)`` is identically ``a·b**x·x**c``,
    measured in data space (``_fit_hoerl``). It needs ``x > 0`` and ``y > 0`` for the two logs,
    and rational refuses the irrational logs/exp, so it drops there like the power/gaussian forms.
  - the saturation ``x/(a·x + b)`` (44.2.11, Michaelis-Menten) and hyperbolic ``1/(a·x + b)``
    (44.2.12) — the last two forms — linearise under RECIPROCALS rather than logs. The
    saturation's double reciprocal is the Lineweaver-Burk line ``1/y = a + b·(1/x)``, affine
    in ``1/x``, so the fit lines ``(1/x, 1/y)`` and reads the INTERCEPT as ``a`` and the SLOPE
    as ``b`` (``_fit_saturation``); the hyperbolic's reciprocal of ``y`` alone is the line
    ``1/y = a·x + b``, so the fit lines ``(x, 1/y)`` and reads the SLOPE as ``a`` and the
    INTERCEPT as ``b`` (``_fit_hyperbolic``, distinct from the additive reciprocal ``a/x + b``
    of 44.2.8). Both measure the error on the ORIGINAL model in data space, and both are EXACT
    in rational (a reciprocal of a rational is rational — no logs/roots), so unlike the
    power/log forms they do NOT drop in rational; only a zero reciprocal (a domain miss — an
    ``x = 0`` / ``y = 0`` datum, or a model denominator ``a·x + b = 0``) drops them.

The genuinely non-linearisable sinusoid ``a*sin(b*x + c) + d`` (44.2.9) is the one form
that keeps an iterative optimise engine: it is non-linear ONLY in the frequency ``b``, so
for any fixed ``b`` its best ``(A, B, d)`` is a linear sub-fit over ``{sin(b·x), cos(b·x),
1}`` — which turns the whole fit into a 1-D minimisation of ``SSR(b)`` over the frequency.
Because that objective is multimodal (local minima at harmonics / aliases of the true
frequency), the frequency is found by a coarse grid scan over a data-derived range followed
by a golden-section refinement (``_fit_sinusoidal`` / ``_search_frequency``), and the final
sub-fit at the refined ``b`` is computed in mode-faithful Value arithmetic so the reported
parameters and error still carry the precision verdict. ``sin`` of a non-zero rational is
irrational, so the sinusoid simply drops in rational mode (like power/log/sqrt). ``fit_all``
fits every form, DROPPING any that cannot fit the data (a domain miss, a degenerate
configuration, or a mode that cannot represent the fit) rather than failing the whole
request (44.3). The survivors are scored by one comparable goodness number — the sum of
squared residuals computed in the active mode (44.4), so it carries the usual
exact/inexact verdict — then RANKED best (least error) first and truncated to the best few
(44.5), the closest fits the caller actually wants.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mcp_abacus.expr.value import Mode, NotRepresentableError, Value

_RATIONAL_FIT_DECIMALS = 12  # rational has no scale of its own; materialise data at this one
_BEST_N = 3  # the fit tool reports the best this-many forms by error (44.5); fewer if fewer fit

# Sinusoidal fit (44.2.9): the lone non-linearisable form, fitted by an iterative frequency
# search rather than in closed form. SSR(b) is multimodal in the frequency b, so a coarse
# grid scan picks the best b before a golden-section refinement polishes it.
_SINUSOID_SCAN_POINTS = 400  # frequencies the multimodal SSR(b) is coarse-scanned over
_SINUSOID_REFINE_ITERATIONS = 100  # golden-section steps polishing the best coarse frequency
_INV_PHI = (5**0.5 - 1) / 2  # 0.618…, 1/golden-ratio — the refinement's interval shrink factor
_SINUSOID_MIN_DISTINCT_X = 4  # a*sin(b*x+c)+d has four parameters; fewer distinct x cannot pin it


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
    any restriction the form places on the data (usually on ``x`` — ``"x > 0"`` for the power
    form — but on ``y`` for the exponential, ``"y > 0"``), or ``None`` when the data is
    unrestricted. ``fit`` is the fitter: it takes the data as
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


def _affine_equation(coeffs: list[Value], terms: "tuple[Callable[[Value], str], ...]") -> str:
    """Render an affine model ``c₀·f₀(x) + c₁·f₁(x) + …`` over ``x`` with clean signs.

    The shared renderer for the affine non-polynomial forms (logarithmic ``a + b*ln(x)``,
    square-root ``a*sqrt(x) + b``, reciprocal ``a/x + b``): each ``terms[j]`` renders the
    unsigned body of its term from a magnitude Value (e.g. ``m -> f"{m}*ln(x)"``), and this
    splits the sign off exactly as :func:`_polynomial_equation` does — the leading term
    keeps its own sign, each later term reads ``+ body`` or ``- |body|`` so the equation
    flows naturally (``a*sqrt(x) - 1`` rather than ``a*sqrt(x) + -1``). The result is valid
    calculate source over ``x`` (44.5), so it pastes straight back in.
    """
    parts = [terms[0](coeffs[0])]  # leading term carries its own sign
    for coeff, term in zip(coeffs[1:], terms[1:], strict=True):
        negative = coeff.to_float() < 0
        magnitude = coeff.neg() if negative else coeff
        parts.append(f"{'-' if negative else '+'} {term(magnitude)}")
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


def _sum_squared_residuals(model: list[Value], ys: list[Value], mode: Mode, scale: int) -> Value:
    """The fit error ``Σ (modelᵢ − yᵢ)²`` in the active mode — the goodness number (44.4)."""
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


def _linear_basis_fitter(
    name: str,
    parameters: tuple[str, ...],
    basis: "tuple[Callable[[Value, Mode, int], Value], ...]",
    render: "Callable[[list[Value]], str]",
) -> "Callable[[list[Value], list[Value], Mode, int], FitResult]":
    """Build the least-squares fitter for a model LINEAR IN ITS PARAMETERS over ``basis``.

    A form is linear in its parameters when it is a weighted sum of fixed basis functions of
    ``x`` — ``model(x) = Σⱼ cⱼ·fⱼ(x)`` — even when each ``fⱼ`` is itself curved (``x**2``,
    ``ln x``, ``1/x``, …). Every such form shares ONE closed-form fit (44.9): its
    least-squares coefficients are the solution of the normal equations ``(Bᵀ·B)·c = Bᵀ·y``,
    where ``B`` is the design matrix ``Bᵢⱼ = fⱼ(xᵢ)``. This generalises the polynomial's
    Vandermonde fit — the polynomial is just the basis ``{x**degree, …, x, 1}`` — so the
    quadratic and cubic (44.2.2 / 44.2.3) and any future linear-in-parameters combo
    (logarithmic ``{1, ln x}``, square-root ``{1, sqrt x}``, reciprocal ``{1, 1/x}``, …) wire
    a basis here instead of hand-rolling their own normal equations.

    ``basis`` lists the functions ``fⱼ(x, mode, scale)`` in the SAME order as ``parameters``
    names their coefficients, so the solved ``c`` lines up with the names and with whatever
    ``render`` produces (the polynomial passes its basis highest-degree first so ``a`` is the
    leading coefficient). Each ``fⱼ`` takes the mode and working scale too, so a basis term
    can mint a mode-faithful constant (the ``1`` term) or a transcendental (``ln x``). The
    matrix ``Bᵀ·B``, the right-hand side ``Bᵀ·y``, the Gauss-Jordan solve and the rebuilt
    model are all accumulated in mode-faithful Value arithmetic, so the whole fit stays exact
    in rational and mode-rounded otherwise. ``render`` turns the solved coefficients into the
    fitted equation string over ``x``.

    The returned fitter has the CurveForm.fit signature. It raises FitError when the data
    underdetermines the basis (a singular ``Bᵀ·B`` — fewer independent points than basis
    functions) via :func:`_solve_linear_system`, and converts a basis function's domain miss
    into a FitError so the form is dropped, not fatal (44.3): a NotRepresentableError (a
    domain miss or an irrational it cannot represent — e.g. ``ln x`` in rational mode, or
    ``√x`` of a negative ``x``) OR a ZeroDivisionError (a ``1/x`` basis at ``x = 0``, which
    is the reciprocal form's domain miss).
    """

    def fit(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
        # Every x equal collapses each design row to the same vector — the basis is
        # underdetermined (rank 1 for ≥2 functions). Detected DIRECTLY here, as in
        # _line_coeffs, rather than via the singular-pivot test below: in fixed-point the
        # normal-equations determinant can round to a tiny NON-zero for identical inputs
        # (the rounded transcendental basis values, ln x / √x, do not cancel exactly), which
        # would otherwise slip a degenerate fit through. The caller guarantees ≥2 points.
        if all(x.to_string() == xs[0].to_string() for x in xs):
            raise FitError(
                f"Cannot fit {name}: every x value is equal, so its basis is degenerate. "
                "Provide data with at least two distinct x values."
            )
        try:
            # Design matrix once: row i holds [f₀(xᵢ), …, f_{k-1}(xᵢ)], reused for both the
            # normal equations and the rebuilt model after the solve.
            design = [[f(x, mode, scale) for f in basis] for x in xs]
        except (NotRepresentableError, ZeroDivisionError) as exc:
            raise FitError(
                f"Cannot fit {name}: the data leaves its basis undefined or unrepresentable "
                f"in {mode.value} mode (a domain miss — e.g. a non-positive log or √, or a "
                f"1/x at x = 0). {exc}"
            ) from exc
        k = len(basis)
        matrix = [
            [_sum([row[j].mul(row[m]) for row in design], mode, scale) for m in range(k)]
            for j in range(k)
        ]
        rhs = [
            _sum([row[j].mul(y) for row, y in zip(design, ys, strict=True)], mode, scale)
            for j in range(k)
        ]
        coeffs = _solve_linear_system(matrix, rhs, name, mode, scale)  # in basis order
        model = [_sum([c.mul(row[j]) for j, c in enumerate(coeffs)], mode, scale) for row in design]
        error = _sum_squared_residuals(model, ys, mode, scale)
        equation = render(coeffs)
        return FitResult(name, equation, tuple(zip(parameters, coeffs, strict=True)), error)

    return fit


def _power_term(power: int) -> "Callable[[Value, Mode, int], Value]":
    """The basis function ``x**power`` for the polynomial fit, built by repeated ``mul``.

    ``power`` 0 is the constant ``1`` term — minted as a mode-faithful Value from the passed
    mode and scale (``x`` may be zero, so it cannot be recovered from ``x`` itself). Higher
    powers multiply ``x`` in, keeping the mode's own rounding at each step.
    """

    def f(x: Value, mode: Mode, scale: int) -> Value:
        result = Value.from_real(1, mode, scale)
        for _ in range(power):
            result = result.mul(x)
        return result

    return f


def _polynomial_fitter(
    name: str, degree: int, parameters: tuple[str, ...]
) -> "Callable[[list[Value], list[Value], Mode, int], FitResult]":
    """Build the fitter for the degree-``degree`` polynomial form (44.2.2 / 44.2.3).

    A polynomial is linear in its coefficients, so it is just the linear-basis fit (44.9)
    over the power basis ``{x**degree, x**(degree-1), …, x, 1}`` — highest degree first, so
    the solved coefficients land highest-degree first, ``a`` the leading coefficient,
    matching ``parameters`` and the equation :func:`_polynomial_equation` renders. The shared
    :func:`_linear_basis_fitter` builds the normal equations ``(Bᵀ·B)·c = Bᵀ·y`` (here ``B``
    the Vandermonde matrix) and raises FitError when the data is degenerate (fewer than
    ``degree + 1`` distinct ``x`` — a singular matrix).
    """
    basis = tuple(_power_term(degree - j) for j in range(degree + 1))  # highest degree first
    return _linear_basis_fitter(name, parameters, basis, _polynomial_equation)


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


def _fit_exponential(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Least-squares fit of the exponential ``a*exp(b*x)`` to the data (44.2.5); domain ``y > 0``.

    The exponential linearises under logs of ``y`` ALONE (``x`` stays raw): ``ln y = ln a +
    b·x`` is a straight line in ``(x, ln y)`` — so the fit is :func:`_line_coeffs` on ``(xs,
    ln ys)``, with ``b`` the slope and ``a = exp(intercept)``, every step in mode-faithful
    Value arithmetic. The error is the sum of squared residuals ``Σ (a·exp(b·xᵢ) − yᵢ)²`` of
    the ORIGINAL model in the active mode, so it measures the fit where the data lives, not
    in log space.

    Raises FitError (so the form is dropped, 44.3) when the logs are undefined or
    unrepresentable: any ``yᵢ ≤ 0`` has no real log, and in rational mode the logs and
    ``exp`` are irrational — the type refuses them — so the exponential form simply does not
    fit there (like the power form). NotRepresentableError from any of those is converted to
    FitError.
    """
    try:
        log_y = [y.ln() for y in ys]
        slope, intercept = _line_coeffs(xs, log_y, mode, scale)  # raw xs, logged ys
        b = slope
        a = intercept.exp()
        model = [a.mul(b.mul(x).exp()) for x in xs]
        error = _sum_squared_residuals(model, ys, mode, scale)
    except NotRepresentableError as exc:
        raise FitError(
            "Cannot fit a*exp(b*x): it needs y > 0, and a mode that can represent the "
            f"logarithms (rational refuses the irrational logs). {exc}"
        ) from exc
    b_str = f"({b.to_string()})" if b.to_float() < 0 else b.to_string()
    equation = f"{a.to_string()}*exp({b_str}*x)"
    return FitResult("exponential", equation, (("a", a), ("b", b)), error)


def _fit_exp_reciprocal(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Least-squares fit of the Arrhenius law ``a*exp(b/x)`` (44.2.14); domain ``x != 0, y > 0``.

    The exponential's cousin with ``1/x`` in place of ``x`` — the ubiquitous rate / vapour-
    pressure shape. It linearises under logs of ``y`` against the RECIPROCAL of ``x``: ``ln y =
    ln a + b·(1/x)`` is a straight line in ``(1/x, ln y)`` — so the fit is :func:`_line_coeffs`
    on ``(1/xs, ln ys)``, with ``b`` the slope and ``a = exp(intercept)``, every step in
    mode-faithful Value arithmetic. The error is the sum of squared residuals
    ``Σ (a·exp(b/xᵢ) − yᵢ)²`` of the ORIGINAL model in the active mode, so it measures the fit
    where the data lives, not in log/reciprocal space (like the exponential fit).

    Raises FitError (so the form is dropped, 44.3) when the transform is undefined or
    unrepresentable: any ``xᵢ = 0`` has no ``1/x`` (ZeroDivisionError), any ``yᵢ ≤ 0`` has no
    real log, and in rational mode the logs and ``exp`` are irrational — the type refuses them
    — so the form simply does not fit there (like the exponential/power forms). An OverflowError
    from a blown-up ``exp`` (a large ``b/x`` near ``x = 0``) drops it too rather than crashing
    the other forms.
    """
    try:
        one = Value.from_real(1, mode, scale)
        inv_x = [one.div(x) for x in xs]
        log_y = [y.ln() for y in ys]
        slope, intercept = _line_coeffs(inv_x, log_y, mode, scale)  # reciprocal xs, logged ys
        b = slope
        a = intercept.exp()
        model = [a.mul(b.mul(rx).exp()) for rx in inv_x]  # a·exp(b·(1/x)) == a·exp(b/x)
        error = _sum_squared_residuals(model, ys, mode, scale)
    except (NotRepresentableError, ZeroDivisionError, OverflowError) as exc:
        raise FitError(
            "Cannot fit a*exp(b/x): it needs x != 0 for the reciprocal and y > 0 for the "
            "logarithm, and a mode that can represent them (rational refuses the irrational "
            f"logs / exp). {exc}"
        ) from exc
    b_str = f"({b.to_string()})" if b.to_float() < 0 else b.to_string()
    equation = f"{a.to_string()}*exp({b_str}/x)"
    return FitResult("exp-reciprocal", equation, (("a", a), ("b", b)), error)


# The affine non-polynomial basis functions (44.2.6-44.2.8): each is fⱼ(x, mode, scale) ->
# Value, wired into _linear_basis_fitter. The constant `1` term mints a mode-faithful Value
# (x may be zero, so it cannot be recovered from x); the others apply the mode's own ln/
# sqrt/reciprocal — a domain miss (non-positive ln/sqrt, 1/x at x=0) or an irrational the
# mode refuses raises, and the fitter catches it to drop the form (44.3).
def _one(x: Value, mode: Mode, scale: int) -> Value:
    return Value.from_real(1, mode, scale)


def _ln_x(x: Value, mode: Mode, scale: int) -> Value:
    return x.ln()


def _sqrt_x(x: Value, mode: Mode, scale: int) -> Value:
    return x.sqrt()


def _recip_x(x: Value, mode: Mode, scale: int) -> Value:
    return Value.from_real(1, mode, scale).div(x)


def _x(x: Value, mode: Mode, scale: int) -> Value:
    return x  # the identity basis term, for the Laurent form's `x` (44.2.13)


def _logarithmic_equation(coeffs: list[Value]) -> str:
    """Render ``a + b*ln(x)`` over ``x`` with clean signs (basis ``{1, ln x}``)."""
    return _affine_equation(coeffs, (lambda m: m.to_string(), lambda m: f"{m.to_string()}*ln(x)"))


def _square_root_equation(coeffs: list[Value]) -> str:
    """Render ``a*sqrt(x) + b`` over ``x`` with clean signs (basis ``{sqrt x, 1}``)."""
    return _affine_equation(coeffs, (lambda m: f"{m.to_string()}*sqrt(x)", lambda m: m.to_string()))


def _reciprocal_equation(coeffs: list[Value]) -> str:
    """Render ``a/x + b`` over ``x`` with clean signs (basis ``{1/x, 1}``)."""
    return _affine_equation(coeffs, (lambda m: f"{m.to_string()}/x", lambda m: m.to_string()))


def _laurent_equation(coeffs: list[Value]) -> str:
    """Render ``a + b*x + c/x`` over ``x`` with clean signs (basis ``{1, x, 1/x}``)."""
    return _affine_equation(
        coeffs,
        (lambda m: m.to_string(), lambda m: f"{m.to_string()}*x", lambda m: f"{m.to_string()}/x"),
    )


# --- Sinusoidal fit (44.2.9) --------------------------------------------------
# The lone non-linearisable form: ``a*sin(b*x + c) + d`` has no closed-form least-squares
# solution. The trick is that it is non-linear ONLY in the frequency ``b`` — expanded as
# ``A*sin(b*x) + B*cos(b*x) + d`` (with ``A = a*cos c``, ``B = a*sin c``) it is LINEAR in
# ``(A, B, d)`` for any fixed ``b``, an ordinary least-squares fit over the basis
# ``{sin(b*x), cos(b*x), 1}``. So the 4-parameter fit reduces to a 1-D search over ``b``:
# for each candidate frequency the best ``(A, B, d)`` is the linear sub-fit, and its residual
# is the objective ``SSR(b)`` to minimise. ``SSR(b)`` is MULTIMODAL (local minima at
# harmonics / aliases of the true frequency), so a single golden-section run would get
# trapped — instead a coarse grid scan across a data-derived frequency range picks the best
# ``b``, then a golden-section refinement polishes it. The scan and refinement run on the
# FLOAT shadow (exactly as the solver's loop drives its search), and only the FINAL sub-fit
# at the refined ``b`` is computed in mode-faithful Value arithmetic, so the reported
# parameters and error carry the active mode's precision verdict (44.9 keeps the closed-form
# forms; this one alone keeps the iterative optimise engine).


def _solve_float(matrix: list[list[float]], rhs: list[float]) -> "list[float] | None":
    """Solve the small square float system ``matrix·c = rhs`` by Gauss-Jordan with pivoting.

    The float-shadow counterpart to :func:`_solve_linear_system`, used inside the frequency
    scan's ``SSR(b)`` objective where speed matters and the active mode does not (the search
    drives on floats, like the solver's loop). Returns ``None`` when the matrix is singular —
    the basis is underdetermined at this frequency — so the caller can penalise that ``b``.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if aug[pivot][col] == 0.0:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot_val
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _sinusoidal_ssr_float(xs: list[float], ys: list[float], b: float) -> float:
    """The frequency search's objective: the least-squares ``SSR`` at a fixed frequency ``b``.

    Builds the linear sub-fit over ``{sin(b*x), cos(b*x), 1}`` on the float shadow — the
    best ``(A, B, d)`` for this ``b`` by the normal equations — and returns the residual sum
    of squares ``Σ (A·sin(b·xᵢ) + B·cos(b·xᵢ) + d − yᵢ)²``. A singular sub-fit (an
    underdetermined basis at this ``b``) is penalised with ``+inf`` so the search steers away.
    """
    n = len(xs)
    sins = [math.sin(b * x) for x in xs]
    coss = [math.cos(b * x) for x in xs]
    cols = (sins, coss, [1.0] * n)
    matrix = [[sum(cols[j][i] * cols[m][i] for i in range(n)) for m in range(3)] for j in range(3)]
    rhs = [sum(cols[j][i] * ys[i] for i in range(n)) for j in range(3)]
    sol = _solve_float(matrix, rhs)
    if sol is None:
        return math.inf
    a_c, b_c, d_c = sol
    return sum((a_c * sins[i] + b_c * coss[i] + d_c - ys[i]) ** 2 for i in range(n))


def _search_frequency(xs: list[float], ys: list[float]) -> float:
    """Find the frequency ``b`` minimising ``SSR(b)`` — coarse scan, then golden-section refine.

    The frequency range is derived from the DATA: ``b`` runs from roughly one slow half-cycle
    over the whole x-span (``b_lo = π/L``, ``L = max x − min x``) up to the sampling's
    Nyquist-ish limit (``b_hi = π/dx``, ``dx`` the smallest gap between distinct sorted x).
    ``SSR(b)`` is scanned on a bounded grid of ``_SINUSOID_SCAN_POINTS`` frequencies — coarse
    enough to be fast and deterministic, fine enough to land in the true minimum's basin —
    and the best grid point's neighbourhood is then polished by a golden-section minimisation
    (the same derivative-free shrink the solver uses), giving the refined frequency.
    """
    distinct = sorted(set(xs))
    span = distinct[-1] - distinct[0]
    smallest_gap = min(hi - lo for lo, hi in zip(distinct, distinct[1:], strict=False))
    b_lo = math.pi / span  # one slow half-cycle over the whole span
    b_hi = math.pi / smallest_gap  # the sampling's Nyquist angular limit
    step = (b_hi - b_lo) / _SINUSOID_SCAN_POINTS
    best_b = b_lo
    best_ssr = math.inf
    for k in range(_SINUSOID_SCAN_POINTS + 1):
        b = b_lo + k * step
        ssr = _sinusoidal_ssr_float(xs, ys, b)
        if ssr < best_ssr:
            best_ssr, best_b = ssr, b
    # Golden-section refine within the grid cell straddling the best coarse frequency.
    lo = max(best_b - step, b_lo)
    hi = min(best_b + step, b_hi)
    c = hi - _INV_PHI * (hi - lo)
    d = lo + _INV_PHI * (hi - lo)
    fc = _sinusoidal_ssr_float(xs, ys, c)
    fd = _sinusoidal_ssr_float(xs, ys, d)
    for _ in range(_SINUSOID_REFINE_ITERATIONS):
        if hi - lo <= 1e-12:
            break
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi - _INV_PHI * (hi - lo)
            fc = _sinusoidal_ssr_float(xs, ys, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _INV_PHI * (hi - lo)
            fd = _sinusoidal_ssr_float(xs, ys, d)
    return (lo + hi) / 2


def _sinusoidal_equation(a: Value, b: Value, c: Value, d: Value) -> str:
    """Render ``a*sin(b*x + c) + d`` over ``x`` with clean signs on ``c`` and ``d``.

    ``a`` is the canonical positive amplitude (``hypot``) and ``b`` the positive frequency,
    so both lead with their own (non-negative) magnitude; the phase ``c`` inside the sine and
    the offset ``d`` read ``+ |·|`` or ``- |·|`` (``3*sin(2*x - 0.5) - 1``, never ``+ -1``),
    the same sign-splitting convention as :func:`_affine_equation`. The result is valid
    calculate source over ``x`` (44.5), so it pastes straight back in.
    """
    c_neg = c.to_float() < 0
    c_mag = (c.neg() if c_neg else c).to_string()
    inner = f"{b.to_string()}*x {'-' if c_neg else '+'} {c_mag}"
    d_neg = d.to_float() < 0
    d_mag = (d.neg() if d_neg else d).to_string()
    return f"{a.to_string()}*sin({inner}) {'-' if d_neg else '+'} {d_mag}"


def _fit_sinusoidal(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Least-squares fit of the sinusoid ``a*sin(b*x + c) + d`` to the data (44.2.9).

    The only form with no closed-form least-squares solution, fitted by an iterative
    frequency search instead. The model is non-linear ONLY in the frequency ``b``: expanded
    as ``A*sin(b*x) + B*cos(b*x) + d`` it is linear in ``(A, B, d)`` for any fixed ``b``, so
    the 4-parameter fit reduces to a 1-D minimisation of ``SSR(b)`` over the frequency
    (:func:`_search_frequency` — a coarse grid scan to dodge the multimodal SSR's local
    minima, then a golden-section refinement). At the refined ``b`` the linear sub-fit over
    ``{sin(b*x), cos(b*x), 1}`` is solved ONCE in mode-faithful Value arithmetic (the normal
    equations via :func:`_solve_linear_system`), and the canonical parameters are recovered:
    ``a = hypot(A, B)`` (the positive amplitude), ``c = atan2(B, A)`` (the phase), ``d`` as
    is. The error is the sum of squared residuals of that model in the active mode, so it
    carries the usual exact/inexact verdict.

    Raises FitError (so the form is DROPPED, 44.3) when it cannot fit: ``sin`` of a non-zero
    rational is irrational, so rational mode refuses the whole form (a NotRepresentableError
    converted here — the sinusoid simply does not fit in rational, like the power/log/sqrt
    forms); and the four parameters need at least four distinct ``x`` to be determined, so
    fewer drops it cleanly. It fits in fixed-point and floating-point.
    """
    try:
        Value.from_real(1, mode, scale).sin()  # probe: rational refuses sin, drop early
    except NotRepresentableError as exc:
        raise FitError(
            "Cannot fit a*sin(b*x + c) + d: it needs a mode that can represent sines, and "
            f"sine of a non-zero rational is irrational (rational refuses it). {exc}"
        ) from exc
    xs_f = [x.to_float() for x in xs]
    ys_f = [y.to_float() for y in ys]
    if len(set(xs_f)) < _SINUSOID_MIN_DISTINCT_X:
        raise FitError(
            f"Cannot fit a*sin(b*x + c) + d: its four parameters need at least "
            f"{_SINUSOID_MIN_DISTINCT_X} distinct x values to be determined."
        )
    frequency = _search_frequency(xs_f, ys_f)
    try:
        b = Value.from_real(frequency, mode, scale)
        sins = [b.mul(x).sin() for x in xs]
        coss = [b.mul(x).cos() for x in xs]
        one = Value.from_real(1, mode, scale)
        design = [[sins[i], coss[i], one] for i in range(len(xs))]
        matrix = [
            [_sum([row[j].mul(row[m]) for row in design], mode, scale) for m in range(3)]
            for j in range(3)
        ]
        rhs = [
            _sum([row[j].mul(y) for row, y in zip(design, ys, strict=True)], mode, scale)
            for j in range(3)
        ]
        big_a, big_b, d = _solve_linear_system(matrix, rhs, "sinusoidal", mode, scale)
        a = big_a.hypot(big_b)  # canonical positive amplitude sqrt(A**2 + B**2)
        c = big_b.atan2(big_a)  # phase atan2(B, A), in (-pi, pi]
        model = [big_a.mul(sins[i]).add(big_b.mul(coss[i])).add(d) for i in range(len(xs))]
        error = _sum_squared_residuals(model, ys, mode, scale)
    except NotRepresentableError as exc:
        raise FitError(
            "Cannot fit a*sin(b*x + c) + d: the data leaves it unrepresentable in "
            f"{mode.value} mode (the sines / amplitude / phase are irrational there). {exc}"
        ) from exc
    equation = _sinusoidal_equation(a, b, c, d)
    return FitResult("sinusoidal", equation, (("a", a), ("b", b), ("c", c), ("d", d)), error)


def _gaussian_equation(a: Value, b: Value, c: Value) -> str:
    """Render ``a*exp(-(x - b)**2/(2*c**2))`` over ``x`` with a clean sign on the centre ``b``.

    The peak ``a`` and width ``c`` are always positive (``a = exp(…)``, ``c = sqrt(positive)``),
    so each leads with its own magnitude; only the centre ``b`` carries a sign, and it is split
    INSIDE the ``(x - b)`` group so a negative centre reads ``(x + |b|)`` rather than the ugly
    ``(x - -2)`` — the same sign-splitting convention as :func:`_sinusoidal_equation` /
    :func:`_affine_equation`. The result is valid calculate source over ``x`` (44.5), so it
    pastes straight back in.
    """
    b_neg = b.to_float() < 0
    b_mag = (b.neg() if b_neg else b).to_string()
    inner = f"x {'+' if b_neg else '-'} {b_mag}"
    return f"{a.to_string()}*exp(-({inner})**2/(2*{c.to_string()}**2))"


def _fit_gaussian(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Closed-form fit of the gaussian ``a*exp(-(x-b)**2/(2*c**2))`` (44.2.10); domain ``y > 0``.

    Fitted in CLOSED FORM by Caruana's method: taking logs turns the bell into a QUADRATIC in
    ``x`` — ``ln y = ln a − (x−b)²/(2c²) = p₂·x² + p₁·x + p₀`` with ``p₂ = −1/(2c²)``,
    ``p₁ = b/c²``, ``p₀ = ln a − b²/(2c²)``. So the fit logs the ``y`` values, least-squares
    fits the quadratic ``{x², x, 1}`` to ``(x, ln y)`` (the normal equations over the power
    basis, solved by :func:`_solve_linear_system` in mode-faithful Value arithmetic), and
    RECOVERS the gaussian parameters from the coefficients ``(p₂, p₁, p₀)``:

      - ``c = sqrt(−1/(2·p₂))`` — the positive width (``−1/(2·p₂) > 0`` because ``p₂ < 0``);
      - ``b = −p₁/(2·p₂)`` — the centre;
      - ``a = exp(p₀ − p₂·b²)`` — the positive peak (since ``b²/(2c²) = −p₂·b²``).

    The recovery needs a DOWNWARD parabola (``p₂ < 0``), checked BEFORE the ``sqrt`` of
    ``−1/(2p₂))``: data whose log-quadratic opens upward (or is flat, ``p₂ ≥ 0``) is not
    bell-shaped, so the form is dropped (a FitError). The error is the sum of squared
    residuals ``Σ (a·exp(−(xᵢ−b)²/(2c²)) − yᵢ)²`` of the model in DATA space (where the data
    lives, not in log space — like the power/exponential fits), in the active mode.

    Raises FitError (so the form is DROPPED, 44.3) when the logs are undefined or
    unrepresentable: any ``yᵢ ≤ 0`` has no real log, and in rational mode the logs / ``exp`` /
    ``sqrt`` are irrational — the type refuses them — so the gaussian simply does not fit
    there (like the power form). NotRepresentableError from any of those is converted here.
    """
    try:
        log_y = [y.ln() for y in ys]
        # Least-squares quadratic over the power basis {x**2, x, 1}, fitted to (xs, log_y):
        # the same normal-equations construction as the sinusoid's linear sub-fit (44.9).
        basis = (_power_term(2), _power_term(1), _power_term(0))
        design = [[f(x, mode, scale) for f in basis] for x in xs]
        matrix = [
            [_sum([row[j].mul(row[m]) for row in design], mode, scale) for m in range(3)]
            for j in range(3)
        ]
        rhs = [
            _sum([row[j].mul(ly) for row, ly in zip(design, log_y, strict=True)], mode, scale)
            for j in range(3)
        ]
        p2, p1, p0 = _solve_linear_system(matrix, rhs, "gaussian", mode, scale)
        if p2.to_float() >= 0:  # not a downward parabola -> not bell-shaped, drop the form
            raise FitError(
                "Cannot fit a*exp(-(x-b)**2/(2*c**2)): the logged data is not a downward "
                "parabola (its quadratic's leading coefficient is not negative), so the data "
                "is not bell-shaped."
            )
        two = Value.from_real(2, mode, scale)
        one = Value.from_real(1, mode, scale)
        c = one.neg().div(two.mul(p2)).sqrt()  # sqrt(-1/(2*p2)), positive (p2 < 0)
        b = p1.neg().div(two.mul(p2))  # -p1/(2*p2)
        a = p0.sub(p2.mul(b.mul(b))).exp()  # exp(p0 - p2*b**2), positive
        two_c2 = two.mul(c.mul(c))  # 2*c**2, the model's denominator
        model = [
            a.mul(x.sub(b).mul(x.sub(b)).neg().div(two_c2).exp()) for x in xs
        ]  # a*exp(-(x-b)**2/(2*c**2)), measured in data space
        error = _sum_squared_residuals(model, ys, mode, scale)
    except (NotRepresentableError, OverflowError) as exc:
        # NotRepresentableError: a non-positive y has no log, or rational refuses the
        # irrational logs/exp/sqrt. OverflowError: near-linear log-data gives a near-flat
        # parabola (p2 -> 0⁻), so the recovered width/peak blow up — an infinitely wide,
        # degenerate bell, dropped like any other unfittable form (44.3).
        raise FitError(
            "Cannot fit a*exp(-(x-b)**2/(2*c**2)): it needs y > 0 and bell-shaped data, and "
            "a mode that can represent the logarithms / exp / sqrt of the fit (rational "
            f"refuses the irrationals). {exc}"
        ) from exc
    equation = _gaussian_equation(a, b, c)
    return FitResult("gaussian", equation, (("a", a), ("b", b), ("c", c)), error)


def _hoerl_equation(a: Value, b: Value, c: Value) -> str:
    """Render ``a*b**x*x**c`` over ``x`` — the Hoerl model (44.2.15).

    The peak ``a`` and base ``b`` are always positive (both ``exp(…)``), so each renders bare;
    only the exponent ``c`` can be negative, and a negative ``c`` is parenthesised as
    ``x**(-0.3)`` so the ``**`` reads cleanly (the same parenthesise-a-negative-exponent
    convention as the power form's ``a*x**b``). ``**`` binds tighter than ``*``, so
    ``a*b**x*x**c`` parses as ``a·(b**x)·(x**c)`` — valid calculate source over ``x`` (44.5),
    pasteable straight back in.
    """
    c_str = f"({c.to_string()})" if c.to_float() < 0 else c.to_string()
    return f"{a.to_string()}*{b.to_string()}**x*x**{c_str}"


def _fit_hoerl(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Closed-form fit of the Hoerl model ``a*b**x*x**c`` (44.2.15); domain ``x > 0, y > 0``.

    A three-parameter rise-then-decay shape (a power law ``x**c`` modulated by an exponential
    ``b**x``). It linearises under logs to a LINE over TWO regressors — ``ln y = ln a +
    (ln b)·x + c·ln x`` — a model linear in its coefficients over the basis ``{1, x, ln x}``.
    So the fit logs the ``y`` values, least-squares fits that basis to ``(x, ln y)`` (the normal
    equations, solved by :func:`_solve_linear_system` in mode-faithful Value arithmetic, exactly
    as the gaussian's Caruana fit does over its own basis), and RECOVERS the parameters from the
    coefficients ``(p₀, p₁, p₂)``: ``a = exp(p₀)`` (the intercept ``ln a``), ``b = exp(p₁)`` (the
    ``x`` slope ``ln b``) and ``c = p₂`` (the ``ln x`` slope, taken as is). The rebuilt model is
    ``exp(p₀ + p₁·x + p₂·ln x)`` — identically ``a·b**x·x**c`` — measured in DATA space, so the
    error is the sum of squared residuals ``Σ (a·b^xᵢ·xᵢ^c − yᵢ)²`` in the active mode (where the
    data lives, not in log space — like the power/exponential/gaussian fits).

    Raises FitError (so the form is DROPPED, 44.3) when the logs are undefined or
    unrepresentable: any ``xᵢ ≤ 0`` has no ``ln x`` and any ``yᵢ ≤ 0`` no ``ln y``, and in
    rational mode the logs and ``exp`` are irrational — the type refuses them — so the form
    simply does not fit there (like the power/exponential/gaussian forms). An OverflowError from
    a blown-up ``exp`` drops it too rather than crashing the other forms.
    """
    # Every x equal collapses the {1, x, ln x} rows to a rank-1 design; detected DIRECTLY here,
    # as in _line_coeffs / _linear_basis_fitter, rather than via the singular-pivot test, because
    # in fixed-point the normal-equations determinant can round to a tiny NON-zero for identical
    # inputs (the rounded ln x do not cancel exactly) and slip a degenerate fit through.
    if all(x.to_string() == xs[0].to_string() for x in xs):
        raise FitError(
            "Cannot fit a*b**x*x**c: every x value is equal, so its basis is degenerate. "
            "Provide data with at least two distinct x values."
        )
    try:
        log_y = [y.ln() for y in ys]
        # Least-squares line over the basis {1, x, ln x}, fitted to (xs, log_y): the same
        # normal-equations construction as the gaussian's log-quadratic (44.9), a different basis.
        basis = (_one, _x, _ln_x)  # coefficients line up as (ln a, ln b, c)
        design = [[f(x, mode, scale) for f in basis] for x in xs]
        matrix = [
            [_sum([row[j].mul(row[m]) for row in design], mode, scale) for m in range(3)]
            for j in range(3)
        ]
        rhs = [
            _sum([row[j].mul(ly) for row, ly in zip(design, log_y, strict=True)], mode, scale)
            for j in range(3)
        ]
        p0, p1, c = _solve_linear_system(matrix, rhs, "hoerl", mode, scale)
        a = p0.exp()  # exp(ln a), positive
        b = p1.exp()  # exp(ln b), positive
        # Rebuild in data space as exp(p0 + p1*x + p2*ln x) == a*b**x*x**c, reusing the logs.
        model = [
            _sum([coef.mul(row[j]) for j, coef in enumerate((p0, p1, c))], mode, scale).exp()
            for row in design
        ]
        error = _sum_squared_residuals(model, ys, mode, scale)
    except (NotRepresentableError, OverflowError) as exc:
        raise FitError(
            "Cannot fit a*b**x*x**c: it needs x > 0 and y > 0 for the logarithms, and a mode "
            f"that can represent them (rational refuses the irrational logs / exp). {exc}"
        ) from exc
    equation = _hoerl_equation(a, b, c)
    return FitResult("hoerl", equation, (("a", a), ("b", b), ("c", c)), error)


def _saturation_equation(a: Value, b: Value) -> str:
    """Render ``x/(a*x + b)`` over ``x`` with clean signs in the denominator.

    The numerator is the bare ``x``; the denominator ``a*x + b`` is the same affine
    ``c₀·x + c₁`` the line renders, so it reuses :func:`_affine_equation` (the leading
    ``a*x`` keeps its own sign, the constant ``b`` reads ``+ |b|`` or ``- |b|``, e.g.
    ``2*x - 3``). The result is valid calculate source over ``x`` (44.5), so it pastes
    straight back in.
    """
    denom = _affine_equation([a, b], (lambda m: f"{m.to_string()}*x", lambda m: m.to_string()))
    return f"x/({denom})"


def _hyperbolic_equation(a: Value, b: Value) -> str:
    """Render ``1/(a*x + b)`` over ``x`` with clean signs in the denominator.

    The numerator is the bare ``1``; the denominator ``a*x + b`` is the same affine line
    the saturation form renders, so it reuses :func:`_affine_equation` (``2*x - 3`` rather
    than ``2*x + -3``). The result is valid calculate source over ``x`` (44.5), so it pastes
    straight back in.
    """
    denom = _affine_equation([a, b], (lambda m: f"{m.to_string()}*x", lambda m: m.to_string()))
    return f"1/({denom})"


def _fit_saturation(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Closed-form fit of the saturation curve ``x/(a*x + b)`` (44.2.11); domain ``x != 0, y != 0``.

    The Michaelis-Menten saturation rises to a plateau ``y -> 1/a``. It linearises under the
    DOUBLE reciprocal (Lineweaver-Burk): ``1/y = a + b·(1/x)`` is a straight line in
    ``(1/x, 1/y)``, AFFINE in ``1/x``. So the fit transforms ``X = 1/x`` and ``Y = 1/y``,
    runs :func:`_line_coeffs` on the transformed data, and reads the parameters off the line:
    the INTERCEPT is ``a`` and the SLOPE is ``b`` (``1/y = a + b·X``). The error is the sum of
    squared residuals ``Σ (xᵢ/(a·xᵢ + b) − yᵢ)²`` of the ORIGINAL model in the active mode, so
    it measures the fit where the data lives, not in reciprocal space (like the power fit).

    This form is EXACT in rational mode — every reciprocal of a rational is rational, no logs
    or roots — so unlike the power/log/sqrt forms it does NOT drop there; only a genuine domain
    miss drops it. Raises FitError (so the form is DROPPED, 44.3) when a reciprocal hits zero:
    any ``xᵢ = 0`` (no ``1/x``), any ``yᵢ = 0`` (no ``1/y``), or a model denominator
    ``a·xᵢ + b = 0`` raises ZeroDivisionError; NotRepresentableError / OverflowError are caught
    too so a stray domain miss drops cleanly rather than crashing the other forms (44.3).
    """
    try:
        one = Value.from_real(1, mode, scale)
        inv_x = [one.div(x) for x in xs]
        inv_y = [one.div(y) for y in ys]
        slope, intercept = _line_coeffs(inv_x, inv_y, mode, scale)
        a = intercept  # 1/y = a + b·(1/x): the intercept is a
        b = slope  # and the slope is b
        model = [x.div(a.mul(x).add(b)) for x in xs]  # x/(a*x + b) in data space
        error = _sum_squared_residuals(model, ys, mode, scale)
    except (NotRepresentableError, ZeroDivisionError, OverflowError) as exc:
        raise FitError(
            "Cannot fit x/(a*x + b): it needs x != 0 and y != 0 for the double reciprocal, "
            f"and a non-zero denominator a*x + b in the model. {exc}"
        ) from exc
    equation = _saturation_equation(a, b)
    return FitResult("saturation", equation, (("a", a), ("b", b)), error)


def _fit_hyperbolic(xs: list[Value], ys: list[Value], mode: Mode, scale: int) -> FitResult:
    """Closed-form fit of the hyperbolic curve ``1/(a*x + b)`` (44.2.12); domain ``y != 0``.

    A smooth ``1/x``-style falloff. It linearises under the reciprocal of ``y`` alone:
    ``1/y = a·x + b`` is a straight line in ``(x, 1/y)``. So the fit transforms ``Y = 1/y``,
    runs :func:`_line_coeffs` on ``(xs, Y)``, and reads the parameters off the line: the SLOPE
    is ``a`` and the INTERCEPT is ``b`` (``1/y = a·x + b``). This is DISTINCT from the
    reciprocal form ``a/x + b`` (44.2.8), which is additive in ``1/x``; here the reciprocal is
    of the whole affine ``a·x + b``. The error is the sum of squared residuals
    ``Σ (1/(a·xᵢ + b) − yᵢ)²`` of the ORIGINAL model in the active mode (like the power fit).

    This form is EXACT in rational mode — reciprocals of rationals are rational, no logs or
    roots — so it does NOT drop there; only a genuine domain miss drops it. Raises FitError (so
    the form is DROPPED, 44.3) when a reciprocal hits zero: any ``yᵢ = 0`` (no ``1/y``) or a
    model denominator ``a·xᵢ + b = 0`` raises ZeroDivisionError; NotRepresentableError /
    OverflowError are caught too so a stray domain miss drops cleanly rather than crashing the
    other forms (44.3).
    """
    try:
        one = Value.from_real(1, mode, scale)
        inv_y = [one.div(y) for y in ys]
        slope, intercept = _line_coeffs(xs, inv_y, mode, scale)  # raw xs, reciprocal ys
        a = slope  # 1/y = a·x + b: the slope is a
        b = intercept  # and the intercept is b
        model = [one.div(a.mul(x).add(b)) for x in xs]  # 1/(a*x + b) in data space
        error = _sum_squared_residuals(model, ys, mode, scale)
    except (NotRepresentableError, ZeroDivisionError, OverflowError) as exc:
        raise FitError(
            "Cannot fit 1/(a*x + b): it needs y != 0 for the reciprocal, and a non-zero "
            f"denominator a*x + b in the model. {exc}"
        ) from exc
    equation = _hyperbolic_equation(a, b)
    return FitResult("hyperbolic", equation, (("a", a), ("b", b)), error)


# The curve library (44.2): one entry per model form, iterated by the fit tool. Each entry
# DECLARES the form — its name, model template over x, free-parameter names and any domain
# limit — and carries the closed-form fitter that estimates its parameters (44.3). Linear,
# quadratic and cubic share the polynomial normal-equations solver (linear via the line's
# direct formula); the logarithmic, square-root and reciprocal forms wire their own affine
# basis into that same solver (44.9); the power and exponential forms linearise under logs.
# The gaussian (44.2.10) is also closed form, linearising under logs to a quadratic in x
# (Caruana's method) whose coefficients recover the peak/centre/width. The sinusoid (44.2.9)
# alone keeps the iterative frequency search. The final two forms — the saturation
# x/(a*x + b) (44.2.11) and hyperbolic 1/(a*x + b) (44.2.12) curves — are closed form via a
# reciprocal-line transform (the double-reciprocal Lineweaver-Burk line 1/y = a + b·(1/x) for
# the saturation, the line 1/y = a·x + b for the hyperbolic). The Laurent form a + b*x + c/x
# (44.2.13) closes with a DIRECT linear basis {1, x, 1/x} (no transform), wired into the same
# _linear_basis_fitter as the affine log/sqrt/reciprocal forms. The exp-reciprocal / Arrhenius
# form a*exp(b/x) (44.2.14) log-linearises like the exponential, but against 1/x — the line
# ln y = ln a + b*(1/x) — so it drops in rational mode like the power/exponential forms. The
# Hoerl model a*b**x*x**c (44.2.15) log-linearises to a line over two regressors, ln y = ln a +
# (ln b)*x + c*ln x over the basis {1, x, ln x} — the gaussian's twin, fitted by the same normal
# equations, then a/b/c recovered by exp — so it too drops in rational mode.
LINEAR = CurveForm("linear", ("a", "b"), "a*x + b", None, _fit_linear)
QUADRATIC = CurveForm(
    "quadratic",
    ("a", "b", "c"),
    "a*x**2 + b*x + c",
    None,
    _polynomial_fitter("quadratic", 2, ("a", "b", "c")),
)
CUBIC = CurveForm(
    "cubic",
    ("a", "b", "c", "d"),
    "a*x**3 + b*x**2 + c*x + d",
    None,
    _polynomial_fitter("cubic", 3, ("a", "b", "c", "d")),
)
POWER = CurveForm("power", ("a", "b"), "a*x**b", "x > 0", _fit_power)
EXPONENTIAL = CurveForm("exponential", ("a", "b"), "a*exp(b*x)", "y > 0", _fit_exponential)
LOGARITHMIC = CurveForm(
    "logarithmic",
    ("a", "b"),
    "a + b*ln(x)",
    "x > 0",
    _linear_basis_fitter("logarithmic", ("a", "b"), (_one, _ln_x), _logarithmic_equation),
)
SQUARE_ROOT = CurveForm(
    "square-root",
    ("a", "b"),
    "a*sqrt(x) + b",
    "x >= 0",
    _linear_basis_fitter("square-root", ("a", "b"), (_sqrt_x, _one), _square_root_equation),
)
RECIPROCAL = CurveForm(
    "reciprocal",
    ("a", "b"),
    "a/x + b",
    "x != 0",
    _linear_basis_fitter("reciprocal", ("a", "b"), (_recip_x, _one), _reciprocal_equation),
)
# The sinusoid (44.2.9) is the lone non-closed-form entry: domain None (no x restriction —
# the rational limitation is a mode drop, not a domain), fitted by the iterative frequency
# search above rather than by the normal equations.
SINUSOIDAL = CurveForm(
    "sinusoidal",
    ("a", "b", "c", "d"),
    "a*sin(b*x + c) + d",
    None,
    _fit_sinusoidal,
)
# The gaussian (44.2.10): closed form via Caruana's method (logs to a quadratic in x), back
# to closed form after the sinusoid. Domain y > 0 (its logs need positive y); rational refuses
# the irrational logs/exp/sqrt, so it drops there too.
GAUSSIAN = CurveForm(
    "gaussian",
    ("a", "b", "c"),
    "a*exp(-(x-b)**2/(2*c**2))",
    "y > 0",
    _fit_gaussian,
)
# The final two forms (44.2.11 / 44.2.12), closed form via a reciprocal-line transform. The
# saturation needs both x != 0 and y != 0 (its double reciprocal), the hyperbolic only y != 0;
# both are exact in rational (reciprocals of rationals are rational), so neither drops there —
# only a zero reciprocal (a domain miss) drops them.
SATURATION = CurveForm(
    "saturation",
    ("a", "b"),
    "x/(a*x + b)",
    "x != 0, y != 0",
    _fit_saturation,
)
HYPERBOLIC = CurveForm(
    "hyperbolic",
    ("a", "b"),
    "1/(a*x + b)",
    "y != 0",
    _fit_hyperbolic,
)
# The Laurent form (44.2.13): a linear trend with a near-origin 1/x pole, a + b*x + c/x. Linear
# in its parameters over the DIRECT basis {1, x, 1/x} (no transform, unlike the power/reciprocal-
# line forms), so it wires straight into _linear_basis_fitter (44.9) like the log/sqrt/reciprocal
# affine forms. Exact in rational (the basis is a constant, x, and a reciprocal of a rational — no
# logs or roots), so it does NOT drop there; only x = 0 (its 1/x pole) drops it, hence domain
# x != 0. The fitter catches that ZeroDivisionError to drop the form rather than crash the rest.
LAURENT = CurveForm(
    "laurent",
    ("a", "b", "c"),
    "a + b*x + c/x",
    "x != 0",
    _linear_basis_fitter("laurent", ("a", "b", "c"), (_one, _x, _recip_x), _laurent_equation),
)
# The exp-reciprocal / Arrhenius form (44.2.14): the exponential's cousin with 1/x in place of
# x, a*exp(b/x) — the rate / vapour-pressure shape. Log-linearises to the line ln y = ln a +
# b*(1/x) over (1/x, ln y), so it drops in rational mode (irrational logs/exp, like the
# exponential/power forms). Domain x != 0 (the reciprocal) and y > 0 (the log).
EXP_RECIPROCAL = CurveForm(
    "exp-reciprocal",
    ("a", "b"),
    "a*exp(b/x)",
    "x != 0, y > 0",
    _fit_exp_reciprocal,
)
# The Hoerl model (44.2.15): a rise-then-decay a*b**x*x**c (a power law modulated by an
# exponential). Log-linearises to a LINE over two regressors, ln y = ln a + (ln b)*x + c*ln x —
# linear in its coefficients over the basis {1, x, ln x} — fitted by the same normal equations as
# the gaussian's log-quadratic, then a = exp(p0), b = exp(p1), c = p2 recovered. Domain x > 0 and
# y > 0 (both logs); drops in rational mode (irrational logs/exp, like the power/gaussian forms).
HOERL = CurveForm(
    "hoerl",
    ("a", "b", "c"),
    "a*b**x*x**c",
    "x > 0, y > 0",
    _fit_hoerl,
)
CURVE_FORMS: tuple[CurveForm, ...] = (
    LINEAR,
    QUADRATIC,
    CUBIC,
    POWER,
    EXPONENTIAL,
    LOGARITHMIC,
    SQUARE_ROOT,
    RECIPROCAL,
    SINUSOIDAL,
    GAUSSIAN,
    SATURATION,
    HYPERBOLIC,
    LAURENT,
    EXP_RECIPROCAL,
    HOERL,
)


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

    Only the best :data:`_BEST_N` survivors are returned (44.5): the data usually fits
    several forms, and the caller wants the few that fit best, not every one. Fewer than
    ``_BEST_N`` come back when fewer forms fit (a polynomial dropped for too few distinct
    ``x``, the power form dropped in rational mode, …).

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
    # Rank best (least error) first; a stable sort keeps registry order on ties (44.5),
    # then keep only the best few — the caller wants the closest fits, not every one.
    results.sort(key=lambda result: result.error.to_float())
    return results[:_BEST_N]
