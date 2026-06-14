# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver tool's engine: pick an objective, then search for the variable's value.

Built up item by item (TODO 31); the strategy vocabulary reworked in TODO 32. The
`objective` argument (find-root / find-minimum / find-maximum) names what the search
looks for; the golden-section engine drives the unknown to it over a bracket.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum

from mcp_abacus.expr.nodes import EvalError, Node
from mcp_abacus.expr.value import Mode, UndefinedVariableError, Value, VariableStore


class SolverError(Exception):
    """A solver request that cannot be honoured — a bad argument or no solution.

    Unlike LexError / ParseError / EvalError it carries no source line: a solver
    failure is about the request (its type, goal, or bracket) or the search outcome,
    not a position in the expression text.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Objective(Enum):
    """What the search looks for — a root, or an extremum in one direction (32.1).

    This single field replaces the pre-32 ``type``/``goal`` pair: ``FIND_ROOT`` is the
    former solve (drive the expression to zero), and ``FIND_MINIMUM`` / ``FIND_MAXIMUM``
    carry the direction the old ``goal`` held. The solver always minimises an internal
    quantity (``fold_objective``); the objective names what the user is after.
    """

    FIND_ROOT = "find-root"  # where the expression equals zero
    FIND_MINIMUM = "find-minimum"  # where the expression is smallest
    FIND_MAXIMUM = "find-maximum"  # where the expression is largest


# Never-surfaced spellings accepted alongside the canonical find-* names (32.2): the
# pre-32 ``solve`` and the ``minimise`` / ``maximise`` direction words, with their
# short and American -ize forms, all resolve to one Objective. Bare ``optimise`` is
# intentionally absent — it named a direction-less extremum, ambiguous now that the
# direction lives in the objective itself (find-minimum vs find-maximum).
_OBJECTIVE_ALIASES: dict[str, Objective] = {
    "solve": Objective.FIND_ROOT,
    "root": Objective.FIND_ROOT,
    "minimise": Objective.FIND_MINIMUM,
    "minimize": Objective.FIND_MINIMUM,
    "min": Objective.FIND_MINIMUM,
    "maximise": Objective.FIND_MAXIMUM,
    "maximize": Objective.FIND_MAXIMUM,
    "max": Objective.FIND_MAXIMUM,
}


def resolve_objective(objective: str | None) -> Objective:
    """Resolve the search objective from the `objective` argument (32.1 / 32.2).

    ``None`` means ``find-root`` — the default, drive the expression to zero. A given
    value must name an objective: the canonical ``find-root`` / ``find-minimum`` /
    ``find-maximum``, or a never-surfaced alias (the pre-32 ``solve``, and
    ``minimise`` / ``maximise`` with their ``min`` / ``max`` and American spellings).
    Anything else is a SolverError listing the canonical names.
    """
    if objective is None:
        return Objective.FIND_ROOT
    try:
        return Objective(objective)
    except ValueError:
        if objective in _OBJECTIVE_ALIASES:
            return _OBJECTIVE_ALIASES[objective]
        valid = ", ".join(o.value for o in Objective)
        raise SolverError(
            f"Unknown objective: {objective!r}. Valid objectives: {valid}."
        ) from None


def fold_objective(value: Value, objective: Objective) -> Value:
    """Fold the expression's Value into the quantity the search drives to its least (32.1).

    The solver always MINIMISES this quantity, so the objective is folded into the value:
      - find-root    -> ``|expr|``; its least is zero, i.e. a root.
      - find-minimum -> ``expr`` unchanged.
      - find-maximum -> ``-expr``; the least of the negation is the greatest of the value.
    The transform stays in the active mode (``abs_`` / ``neg``), so the search
    compares candidates in the mode's own representation, not a float shadow.
    """
    if objective is Objective.FIND_ROOT:
        return value.abs_()
    if objective is Objective.FIND_MAXIMUM:
        return value.neg()
    return value


def validate_bracket(lower: float, upper: float) -> None:
    """Check the search bracket is a non-empty interval, or raise SolverError (31.6).

    The unknown is searched within ``[lower, upper]``, so ``lower`` must be strictly
    below ``upper`` — the bracket needs width to search. (Whether the unknown actually
    occurs in the expression is a separate check, ``validate_unknown``.)
    """
    if lower >= upper:
        raise SolverError(
            f"Search bracket is empty: lower ({lower}) must be below upper ({upper})."
        )


def validate_unknown(node: Node, variable: str) -> None:
    """Check `variable` is a usable unknown in the parsed program, or raise SolverError (31.4).

    The unknown the solver drives must occur as a REFERENCE in the expression (else
    the search has nothing to vary) and must NOT be an ASSIGNMENT TARGET (a name the
    program binds is a computed constant, not free to vary). Every OTHER name is a
    constant the program is expected to set; a name that is neither the unknown nor
    assigned is left to surface later, at evaluation, as an EvalError — the same
    unset-name error calculate reports.
    """
    if variable in node.assigned_names():
        raise SolverError(
            f"Variable {variable!r} is assigned by the expression, so it is a "
            f"computed constant, not a free unknown to solve for."
        )
    if variable not in node.referenced_names():
        raise SolverError(
            f"Variable {variable!r} does not occur in the expression, so there is "
            f"nothing to solve for."
        )


# --- the search engine (31.7) -------------------------------------------------
# ONE minimizer drives every case: the objective folds find-root/find-maximum into a
# quantity whose LEAST is the answer (fold_objective(), 32.1), so the engine only ever
# minimises. Golden-section search needs no derivative — it brackets the minimum
# and shrinks the interval by the golden ratio each step, evaluating the program
# once per new point. A real (float) search drives the abacus engine, which works
# in the active mode: each candidate is materialised as a mode-faithful Value, the
# program evaluated, and its Value reduced back to a float to compare.

_ALGORITHM = "golden-section-search"  # the one search method today (32.3); the reply
# reports it so a future second algorithm is distinguishable without a shape change.
_INV_PHI = (5**0.5 - 1) / 2  # 0.618..., 1/golden-ratio — the interval shrink factor
_MAX_ITERATIONS = 200  # cap: ~60 steps already shrink a unit bracket below 1e-12
_TIME_LIMIT_SECONDS = 2.0  # hard wall-clock cap: a pathological program can make a
# single candidate evaluation slow, so the iteration cap alone is not enough to bound
# the search — stop after this long and report the best candidate reached so far.
_FLOAT_X_TOL = 1e-12  # float bracket-width stop — a double resolves no finer near 1
_FLOAT_RESIDUAL_TOL = 1e-6  # float solve acceptance: |expr| this small counts as a root
_RATIONAL_SEARCH_DECIMALS = 12  # rational has no scale of its own; search at this one


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The outcome of a successful search — the found unknown and the value there (31.8).

    ``solution`` is the unknown's found value as a mode-faithful Value (the search
    is approximate, so it is the best estimate within tolerance); ``value`` is the
    EXPRESSION's Value evaluated at that solution (for find-root, near zero; for an
    extremum, the extremum). ``objective`` echoes the resolved objective (32.1),
    ``algorithm`` names the search method used (32.3), and ``iterations`` is how many
    interval-shrink steps the search took. The server turns this into the reply dict;
    a failure (bad request, no solution) is a SolverError instead, never a SolverResult.
    """

    variable: str
    objective: Objective
    algorithm: str
    solution: Value
    value: Value
    iterations: int


def _search_scale(node: Node, mode: Mode, floor: int) -> int:
    """The decimal scale a candidate is materialised at, per mode (31.7).

    Fixed-point binds the unknown at the run's own working scale — the floor
    (min_fixed_point_precision) raised to the widest literal scale — so the
    candidate carries the same precision as the constants it is combined with and
    the max()-of-scales propagation keeps it. Rational has no inherent scale, so it
    searches at a fixed working precision. Floating-point ignores the scale (a
    double has none); 0 is returned as an unused placeholder.
    """
    match mode:
        case Mode.FIXED_POINT:
            return max(floor, node._max_literal_scale())
        case Mode.RATIONAL:
            return _RATIONAL_SEARCH_DECIMALS
        case Mode.FLOATING_POINT:
            return 0
        case _:
            raise ValueError(f"unsupported mode: {mode!r}")


def _tolerances(mode: Mode, scale: int) -> tuple[float, float]:
    """``(x_tol, residual_tol)`` for the bracket-width stop and solve acceptance (31.7).

    ``x_tol`` ends the search once the bracket is narrower than the mode can
    resolve; ``residual_tol`` is how close to zero |expr| must come for a solve to
    count as solved. Fixed-point and rational derive both from their scale — one
    unit in the last place, ``10**-scale`` — so the search stops at the grid and an
    exact root (residual 0) is accepted while an unrepresentable one is reported as
    no-solution. Floating-point uses a small absolute epsilon for the bracket and a
    looser residual, since golden-section narrows the UNKNOWN, and |expr| near a
    root is only as small as the local slope times that width.
    """
    match mode:
        case Mode.FLOATING_POINT:
            return _FLOAT_X_TOL, _FLOAT_RESIDUAL_TOL
        case Mode.FIXED_POINT | Mode.RATIONAL:
            unit = 10.0**-scale
            return unit, unit
        case _:
            raise ValueError(f"unsupported mode: {mode!r}")


def search(
    node: Node,
    variable: str,
    lower: float,
    upper: float,
    mode: Mode,
    floor: int,
    objective: Objective,
) -> SolverResult:
    """Golden-section search for the unknown over ``[lower, upper]`` (31.7).

    Minimises the objective (fold_objective(), 32.1) — ``|expr|`` for find-root,
    ``±expr`` for an extremum — by repeatedly evaluating the program with the unknown
    bound to a candidate and shrinking the bracket toward the smaller end. Each candidate is
    materialised in ``mode`` at the working scale, bound into a fresh seeded store,
    and the program evaluated; the resulting Value is reduced to a float to drive the
    search. A candidate that raises a DOMAIN error (e.g. sqrt of a negative) is
    penalised with +inf so the search steers away, but a STRUCTURAL failure — a
    constant the program never sets — propagates as an EvalError (it fails at every
    point, and is the user's to fix, not a region to avoid).

    The search is also bounded by a hard wall-clock limit of ``_TIME_LIMIT_SECONDS``
    (2s): the iteration cap bounds the NUMBER of evaluations, but a single pathological
    candidate can be slow, so the elapsed time is checked each step and the search
    stops once the limit is passed, reporting the best candidate reached so far.

    Returns the best candidate found as a SolverResult. Raises SolverError when the
    expression evaluates nowhere in the bracket, or when a solve cannot drive |expr|
    within ``residual_tol`` of zero (reporting the closest it reached) — including
    when the time limit cut the search short before it could.
    """
    scale = _search_scale(node, mode, floor)
    x_tol, residual_tol = _tolerances(mode, scale)
    deadline = time.monotonic() + _TIME_LIMIT_SECONDS
    # The smallest objective seen and the candidate/value that produced it. Tracking
    # the best across ALL evaluations (not just the final midpoint) keeps the answer
    # honest even if quantisation makes the very last point a hair worse.
    best_obj = math.inf
    best_solution: Value | None = None
    best_value: Value | None = None

    def evaluate_objective(x: float) -> float:
        nonlocal best_obj, best_solution, best_value
        candidate = Value.from_real(x, mode, scale)
        store = VariableStore()
        store.set(variable, candidate)
        try:
            raw = node.evaluate(mode, floor, variables=store)
        except EvalError as exc:
            if isinstance(exc.__cause__, UndefinedVariableError):
                raise  # a constant the program never set — structural, surface it
            return math.inf  # a domain error at THIS candidate — steer the search away
        obj = fold_objective(raw, objective).to_float()
        if obj < best_obj:
            best_obj, best_solution, best_value = obj, candidate, raw
        return obj

    a, b = lower, upper
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc = evaluate_objective(c)
    fd = evaluate_objective(d)
    iterations = 0
    timed_out = False
    while (b - a) > x_tol and iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        if fc <= fd:
            b, d, fd = d, c, fc  # minimum is left of d; reuse c as the new d
            c = b - _INV_PHI * (b - a)
            fc = evaluate_objective(c)
        else:
            a, c, fc = c, d, fd  # minimum is right of c; reuse d as the new c
            d = a + _INV_PHI * (b - a)
            fd = evaluate_objective(d)
        iterations += 1
    if not timed_out:
        evaluate_objective((a + b) / 2)  # the converged midpoint, folded into the best

    # Grid polish (fixed-point / rational): the continuous search stops within a
    # bracket narrower than one grid step, so the best candidate may sit one step
    # off an EXACTLY representable root (the float midpoint quantised to the wrong
    # side). Re-test the grid neighbours of the best point so a root on the grid is
    # found exactly (residual 0) rather than rejected as a hair-too-large miss.
    # Skipped on timeout — the hard cap is already spent, no budget for extra probes.
    if best_solution is not None and mode is not Mode.FLOATING_POINT and not timed_out:
        step = 10.0**-scale
        centre = best_solution.to_float()
        for k in (-2, -1, 1, 2):
            evaluate_objective(centre + k * step)

    if best_solution is None or best_value is None:
        limit = (
            f" within the {_TIME_LIMIT_SECONDS:g}s time limit" if timed_out else ""
        )
        raise SolverError(
            f"The expression could not be evaluated anywhere in [{lower}, {upper}]"
            f"{limit} (every candidate for {variable!r} raised a domain error)."
        )
    if objective is Objective.FIND_ROOT and best_obj > residual_tol:
        limit = (
            f" The search stopped at the {_TIME_LIMIT_SECONDS:g}s time limit."
            if timed_out
            else ""
        )
        raise SolverError(
            f"No solution: the expression does not reach zero for {variable!r} in "
            f"[{lower}, {upper}]. The closest is |expr| = {best_obj:.6g} "
            f"at {variable} = {best_solution.to_string()}.{limit}"
        )
    return SolverResult(variable, objective, _ALGORITHM, best_solution, best_value, iterations)
