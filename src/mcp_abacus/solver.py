# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver tool's engine: pick an objective, then search for the variable's value.

Built up item by item (TODO 31); the strategy vocabulary reworked in TODO 32. The
`objective` argument (find-root / find-minimum / find-maximum) names WHAT the search
looks for; the `algorithm` argument names HOW it searches. Two engines today (TODO
33): golden-section search drives ONE unknown over a bracket, and Nelder-Mead
(33.14) walks a simplex over n unknowns — multivariate, bounds-clamped to each
bracket. Both minimise the same folded objective, so every objective works on either.
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


class Algorithm(Enum):
    """HOW the search drives the unknown(s) — the engine used (TODO 33).

    Split from ``Objective`` (WHAT it looks for): the same root / extremum can be
    reached by different methods. ``GOLDEN_SECTION`` is the single-variable bracket
    shrinker (31.7); ``BRENT_PARABOLIC`` is the single-variable parabolic minimiser
    (33.12), a faster sibling that fits a parabola through the best three points and
    falls back to a golden-section step; ``NELDER_MEAD`` is the multivariate downhill
    simplex (33.14). The enum value is the string reported in the reply's
    ``algorithm`` field (32.3).
    """

    GOLDEN_SECTION = "golden-section-search"  # one unknown, shrink a bracket
    BRENT_PARABOLIC = "brent-parabolic"  # one unknown, parabola + golden fallback
    NELDER_MEAD = "nelder-mead"  # n unknowns, walk a simplex downhill


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
        raise SolverError(f"Unknown objective: {objective!r}. Valid objectives: {valid}.") from None


# Never-surfaced spellings for the `algorithm` argument (the 23.6 alias rule): a short
# / spaced form per engine resolving to one Algorithm. Mirrors _OBJECTIVE_ALIASES.
_ALGORITHM_ALIASES: dict[str, Algorithm] = {
    "golden-section": Algorithm.GOLDEN_SECTION,
    "golden": Algorithm.GOLDEN_SECTION,
    "brent": Algorithm.BRENT_PARABOLIC,
    "parabolic": Algorithm.BRENT_PARABOLIC,
    "nelder mead": Algorithm.NELDER_MEAD,
    "simplex": Algorithm.NELDER_MEAD,
    "downhill-simplex": Algorithm.NELDER_MEAD,
}


def resolve_algorithm(algorithm: str | None) -> Algorithm:
    """Resolve the search engine from the `algorithm` argument (33.14).

    ``None`` means ``golden-section-search`` — the default single-variable engine, so
    every pre-33 call is unchanged. A given value must name an engine: the canonical
    ``golden-section-search`` / ``nelder-mead``, or a never-surfaced alias (``golden``,
    ``simplex``, …). Anything else is a SolverError listing the canonical names.
    """
    if algorithm is None:
        return Algorithm.GOLDEN_SECTION
    try:
        return Algorithm(algorithm)
    except ValueError:
        if algorithm in _ALGORITHM_ALIASES:
            return _ALGORITHM_ALIASES[algorithm]
        valid = ", ".join(a.value for a in Algorithm)
        raise SolverError(f"Unknown algorithm: {algorithm!r}. Valid algorithms: {valid}.") from None


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


# --- the search engines (31.7 golden-section, 33.14 Nelder-Mead) --------------
# Each engine MINIMISES: the objective folds find-root/find-maximum into a quantity
# whose LEAST is the answer (fold_objective(), 32.1), so neither engine needs to know
# the objective beyond that fold. Both are derivative-free and drive the abacus engine
# in the active mode — each candidate is materialised as a mode-faithful Value, the
# program evaluated, and its Value reduced back to a float to compare. Golden-section
# shrinks a 1-D bracket; Nelder-Mead walks an n-vertex simplex over n unknowns. The
# reply names which ran (Algorithm, 32.3/33.14) so the two are distinguishable.

_INV_PHI = (5**0.5 - 1) / 2  # 0.618..., 1/golden-ratio — the interval shrink factor
_GOLDEN = (3 - 5**0.5) / 2  # 0.382..., the complementary golden fraction — Brent's
# fallback step takes this share of the larger sub-bracket when the parabola is rejected
_MAX_ITERATIONS = 200  # cap: ~60 steps already shrink a unit bracket below 1e-12
_TIME_LIMIT_SECONDS = 2.0  # hard wall-clock cap: a pathological program can make a
# single candidate evaluation slow, so the iteration cap alone is not enough to bound
# the search — stop after this long and report the best candidate reached so far.
_FLOAT_X_TOL = 1e-12  # float bracket-width stop — a double resolves no finer near 1
_FLOAT_RESIDUAL_TOL = 1e-6  # float solve acceptance: |expr| this small counts as a root
_RATIONAL_SEARCH_DECIMALS = 12  # rational has no scale of its own; search at this one


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The outcome of a successful search — the found unknown(s) and the value there (31.8).

    ``solutions`` is the ordered ``(name, found Value)`` for every unknown the search
    drove — one entry for golden-section, n for Nelder-Mead (33.14); each found Value
    is a mode-faithful best estimate within tolerance. ``variable`` / ``solution`` echo
    the FIRST (or only) unknown, a convenience for the single-variable case so the 1-D
    reply and the engine tests need not unpack the list. ``value`` is the EXPRESSION's
    Value evaluated at that solution (for find-root, near zero; for an extremum, the
    extremum). ``objective`` echoes the resolved objective (32.1), ``algorithm`` names
    the engine used (32.3/33.14), and ``iterations`` is how many search steps it took.
    The server turns this into the reply dict; a failure (bad request, no solution) is
    a SolverError instead, never a SolverResult.
    """

    variable: str
    objective: Objective
    algorithm: str
    solution: Value
    value: Value
    iterations: int
    solutions: tuple[tuple[str, Value], ...]


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
        limit = f" within the {_TIME_LIMIT_SECONDS:g}s time limit" if timed_out else ""
        raise SolverError(
            f"The expression could not be evaluated anywhere in [{lower}, {upper}]"
            f"{limit} (every candidate for {variable!r} raised a domain error)."
        )
    if objective is Objective.FIND_ROOT and best_obj > residual_tol:
        limit = (
            f" The search stopped at the {_TIME_LIMIT_SECONDS:g}s time limit." if timed_out else ""
        )
        raise SolverError(
            f"No solution: the expression does not reach zero for {variable!r} in "
            f"[{lower}, {upper}]. The closest is |expr| = {best_obj:.6g} "
            f"at {variable} = {best_solution.to_string()}.{limit}"
        )
    return SolverResult(
        variable,
        objective,
        Algorithm.GOLDEN_SECTION.value,
        best_solution,
        best_value,
        iterations,
        ((variable, best_solution),),
    )


# --- Brent's parabolic minimiser (33.12) --------------------------------------
# The single-variable peer of golden-section, usually faster: instead of always
# trisecting the bracket by the golden ratio it fits a parabola through the best
# three points seen and leaps near its vertex, dropping back to a golden step only
# when the parabola is unhelpful (vertex outside the bracket, or too large a move).
# Everything around the core — candidate materialisation, best-tracking, grid polish,
# the time / iteration caps, and the error paths — is the same as golden-section, so
# the two engines differ only in how they pick the next point to evaluate.


def brent_parabolic(
    node: Node,
    variable: str,
    lower: float,
    upper: float,
    mode: Mode,
    floor: int,
    objective: Objective,
) -> SolverResult:
    """Brent's parabolic minimiser for the unknown over ``[lower, upper]`` (33.12).

    A faster-converging sibling of :func:`search`: it minimises the SAME folded
    objective (fold_objective(), 32.1) — ``|expr|`` for find-root, ``±expr`` for an
    extremum — and shares :func:`search`'s machinery exactly (candidate eval in the
    active mode, best-across-all tracking, grid polish, the 2-second wall-clock and
    iteration caps, and the no-evaluation / no-solution error paths). The two differ
    only in HOW the next point is chosen: Brent fits a parabola through the best three
    points and jumps near its vertex, so on a smooth extremum it converges in far fewer
    evaluations than golden-section's fixed trisection.

    The loop is the textbook bounded Brent (Numerical Recipes ``brent`` / SciPy
    ``fminbound``), kept inside ``[a, b]``: the parabolic step is accepted only when
    its vertex lands inside the current bracket and the move is below half the
    step-before-last (the ``e`` bookkeeping); otherwise a golden-section step
    (``_GOLDEN`` of the larger sub-bracket) is taken. A non-smooth objective — the
    kinked ``|expr|`` of a find-root — simply triggers the golden fallback more often,
    so it still converges. Returns the best candidate as a SolverResult; raises
    SolverError on the same conditions as :func:`search`.
    """
    scale = _search_scale(node, mode, floor)
    x_tol, residual_tol = _tolerances(mode, scale)
    deadline = time.monotonic() + _TIME_LIMIT_SECONDS
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

    # x is the best point so far, w the second best, v the previous w; the parabola is
    # fitted through the three. d is the last step, e the step before it (the parabola
    # is only trusted when it asks for less than half of e). Seed all three at one
    # interior point, a golden fraction in from the lower end.
    a, b = lower, upper
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = evaluate_objective(x)
    d = e = 0.0
    iterations = 0
    timed_out = False
    while iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        midpoint = (a + b) / 2
        # Brent's own convergence test: stop once the best point x sits centred within
        # a bracket narrower than the tolerance. Tied to the x_tol minimal step below
        # (sample no closer than x_tol to x), so the search settles instead of taking
        # ever-tinier steps near a smooth minimum the way a bare bracket-width test would.
        if abs(x - midpoint) <= 2 * x_tol - 0.5 * (b - a):
            break
        use_parabola = False
        if abs(e) > x_tol:
            # Fit the parabola through (x, fx), (w, fw), (v, fv); p/q is the step from
            # x to its vertex. Trust it only inside (a, b) and below half the prior e.
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            prev_e, e = e, d
            if abs(p) < abs(0.5 * q * prev_e) and a - x < p / q < b - x:
                d = p / q
                use_parabola = True
        if not use_parabola:  # golden-section fallback into the larger sub-bracket
            e = (b - x) if x < midpoint else (a - x)
            d = _GOLDEN * e
        # Never sample closer than x_tol to x (a zero-width step stalls the search).
        if abs(d) < x_tol:
            d = x_tol if d > 0 else -x_tol
        u = min(max(x + d, a), b)
        fu = evaluate_objective(u)
        if fu <= fx:  # new best — it brackets one side; x slides to u
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:  # u is worse than the best — it tightens the bracket toward x
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
        iterations += 1

    # Grid polish (fixed-point / rational), identical to golden-section's: re-test the
    # grid neighbours of the best point so a root sitting exactly on the grid is found
    # (residual 0) rather than missed by a quantised hair. Skipped on float and timeout.
    if best_solution is not None and mode is not Mode.FLOATING_POINT and not timed_out:
        step = 10.0**-scale
        centre = best_solution.to_float()
        for k in (-2, -1, 1, 2):
            evaluate_objective(centre + k * step)

    if best_solution is None or best_value is None:
        limit = f" within the {_TIME_LIMIT_SECONDS:g}s time limit" if timed_out else ""
        raise SolverError(
            f"The expression could not be evaluated anywhere in [{lower}, {upper}]"
            f"{limit} (every candidate for {variable!r} raised a domain error)."
        )
    if objective is Objective.FIND_ROOT and best_obj > residual_tol:
        limit = (
            f" The search stopped at the {_TIME_LIMIT_SECONDS:g}s time limit." if timed_out else ""
        )
        raise SolverError(
            f"No solution: the expression does not reach zero for {variable!r} in "
            f"[{lower}, {upper}]. The closest is |expr| = {best_obj:.6g} "
            f"at {variable} = {best_solution.to_string()}.{limit}"
        )
    return SolverResult(
        variable,
        objective,
        Algorithm.BRENT_PARABOLIC.value,
        best_solution,
        best_value,
        iterations,
        ((variable, best_solution),),
    )


# --- Nelder-Mead downhill simplex (33.14) -------------------------------------
# The multivariate peer of golden-section: instead of shrinking a 1-D bracket it walks
# a simplex of n+1 vertices over n unknowns, reflecting the worst vertex through the
# centroid of the rest and expanding / contracting / shrinking from there. No
# derivative, one program evaluation per trial point — the same evaluate-fold-compare
# loop, just with a VECTOR bound into the store. Every trial point is clamped to the
# per-axis brackets, so the search stays inside the box the caller gave.

_NM_REFLECT = 1.0  # α — reflect the worst vertex through the centroid
_NM_EXPAND = 2.0  # γ — push further when reflection found a new best
_NM_CONTRACT = 0.5  # ρ — pull back toward the centroid when reflection was poor
_NM_SHRINK = 0.5  # σ — shrink every vertex toward the best when contraction failed
_NM_INIT_STEP = 0.4  # initial per-axis vertex offset, as a fraction of bracket width:
# from the midpoint this lands at lower+0.9·width — a wide, in-box starting simplex.


def nelder_mead(
    node: Node,
    unknowns: list[tuple[str, float, float]],
    mode: Mode,
    floor: int,
    objective: Objective,
) -> SolverResult:
    """Nelder-Mead simplex search for n unknowns over their brackets (33.14).

    ``unknowns`` is the ordered ``(name, lower, upper)`` for each free variable; the
    simplex starts at the per-axis midpoints (plus one vertex offset along each axis)
    and walks downhill on the folded objective (fold_objective(), 32.1) — ``|expr|``
    for find-root, ``±expr`` for an extremum — exactly the quantity golden-section
    minimises, so every objective works here too. Each trial point binds all n names
    into a fresh store, evaluates the program, and reduces the result to a float; a
    point that raises a DOMAIN error is penalised with +inf so the simplex steers
    away, while a STRUCTURAL failure (a constant the program never set) propagates as
    an EvalError. Every trial point is clamped to its ``[lower, upper]``.

    Bounded, like golden-section, by ``_MAX_ITERATIONS`` and the hard 2-second
    wall-clock (``_TIME_LIMIT_SECONDS``); on timeout it stops with the best vertex
    reached so far. Returns a SolverResult whose ``solutions`` lists every unknown's
    found value. Raises SolverError when the program evaluates nowhere in the box, or
    when a find-root cannot drive |expr| within ``residual_tol`` of zero (reporting the
    closest point reached) — including when the time limit cut the search short.
    """
    names = [name for name, _, _ in unknowns]
    lowers = [float(lo) for _, lo, _ in unknowns]
    uppers = [float(hi) for _, _, hi in unknowns]
    n = len(unknowns)
    scale = _search_scale(node, mode, floor)
    x_tol, residual_tol = _tolerances(mode, scale)
    deadline = time.monotonic() + _TIME_LIMIT_SECONDS

    best_obj = math.inf
    best_point: list[float] | None = None
    best_solution: tuple[Value, ...] | None = None
    best_value: Value | None = None

    def clamp(point: list[float]) -> list[float]:
        return [min(max(point[i], lowers[i]), uppers[i]) for i in range(n)]

    def evaluate_objective(point: list[float]) -> float:
        nonlocal best_obj, best_point, best_solution, best_value
        candidates = tuple(Value.from_real(point[i], mode, scale) for i in range(n))
        store = VariableStore()
        for name, candidate in zip(names, candidates, strict=True):
            store.set(name, candidate)
        try:
            raw = node.evaluate(mode, floor, variables=store)
        except EvalError as exc:
            if isinstance(exc.__cause__, UndefinedVariableError):
                raise  # a constant the program never set — structural, surface it
            return math.inf  # a domain error at THIS point — steer the simplex away
        obj = fold_objective(raw, objective).to_float()
        if obj < best_obj:
            best_obj = obj
            best_point, best_solution, best_value = list(point), candidates, raw
        return obj

    def along(centroid: list[float], coeff: float, target: list[float]) -> list[float]:
        # The point `coeff` of the way from the centroid toward `target`, clamped to
        # the box. Reflection/expansion/contraction are all this move at different
        # coefficients (reflection toward the worst with a negative coeff).
        return clamp([centroid[i] + coeff * (target[i] - centroid[i]) for i in range(n)])

    # Initial simplex: the midpoints, plus one vertex per axis offset along that axis.
    midpoint = [(lowers[i] + uppers[i]) / 2 for i in range(n)]
    simplex = [list(midpoint)]
    for i in range(n):
        vertex = list(midpoint)
        vertex[i] = midpoint[i] + _NM_INIT_STEP * (uppers[i] - lowers[i])
        simplex.append(vertex)
    fvals = [evaluate_objective(v) for v in simplex]

    iterations = 0
    timed_out = False
    while iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        order = sorted(range(n + 1), key=lambda k: fvals[k])  # best (least) first
        simplex = [simplex[k] for k in order]
        fvals = [fvals[k] for k in order]
        size = max(max(v[i] for v in simplex) - min(v[i] for v in simplex) for i in range(n))
        if size <= x_tol:  # the simplex has collapsed below what the mode resolves
            break
        centroid = [sum(simplex[k][i] for k in range(n)) / n for i in range(n)]
        worst = simplex[n]
        reflected = along(centroid, -_NM_REFLECT, worst)  # away from the worst vertex
        fr = evaluate_objective(reflected)
        if fvals[0] <= fr < fvals[n - 1]:
            simplex[n], fvals[n] = reflected, fr  # a middling reflection: take it
        elif fr < fvals[0]:  # a new best — try stepping further out
            expanded = along(centroid, _NM_EXPAND, reflected)
            fe = evaluate_objective(expanded)
            if fe < fr:
                simplex[n], fvals[n] = expanded, fe
            else:
                simplex[n], fvals[n] = reflected, fr
        else:  # reflection no better than the second-worst — contract
            if fr < fvals[n]:  # outside contraction (reflection beat the old worst)
                contracted = along(centroid, _NM_CONTRACT, reflected)
            else:  # inside contraction (toward the worst)
                contracted = along(centroid, _NM_CONTRACT, worst)
            fc = evaluate_objective(contracted)
            if fc < fvals[n]:
                simplex[n], fvals[n] = contracted, fc
            else:  # contraction failed too — shrink the whole simplex toward the best
                best_vertex = simplex[0]
                for k in range(1, n + 1):
                    simplex[k] = along(best_vertex, _NM_SHRINK, simplex[k])
                    fvals[k] = evaluate_objective(simplex[k])
        iterations += 1

    # Grid polish (fixed-point / rational), per axis: the simplex stops within a box
    # narrower than one grid step, so the best vertex may sit one step off an EXACTLY
    # representable root. Probe each axis's grid neighbours of the best point — 4·n
    # probes, not the 3^n of a full neighbourhood — so a root on the grid is found
    # exactly. Skipped on float (no grid) and on timeout (the hard cap is spent).
    if best_point is not None and mode is not Mode.FLOATING_POINT and not timed_out:
        step = 10.0**-scale
        centre = list(best_point)
        for i in range(n):
            for k in (-2, -1, 1, 2):
                probe = list(centre)
                probe[i] = centre[i] + k * step
                evaluate_objective(clamp(probe))

    box = ", ".join(f"{names[i]} in [{lowers[i]}, {uppers[i]}]" for i in range(n))
    if best_solution is None or best_value is None:
        limit = f" within the {_TIME_LIMIT_SECONDS:g}s time limit" if timed_out else ""
        raise SolverError(
            f"The expression could not be evaluated anywhere in the search box "
            f"({box}){limit} (every candidate raised a domain error)."
        )
    solutions = tuple(zip(names, best_solution, strict=True))
    if objective is Objective.FIND_ROOT and best_obj > residual_tol:
        limit = (
            f" The search stopped at the {_TIME_LIMIT_SECONDS:g}s time limit." if timed_out else ""
        )
        point = ", ".join(f"{name} = {value.to_string()}" for name, value in solutions)
        raise SolverError(
            f"No solution: the expression does not reach zero in the search box "
            f"({box}). The closest is |expr| = {best_obj:.6g} at {point}.{limit}"
        )
    first_name, first_value = solutions[0]
    return SolverResult(
        first_name,
        objective,
        Algorithm.NELDER_MEAD.value,
        first_value,
        best_value,
        iterations,
        solutions,
    )
