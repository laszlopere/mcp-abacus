# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver tool's engine: pick an objective, then search for the variable's value.

Built up item by item (TODO 31); the strategy vocabulary reworked in TODO 32. The
`objective` argument (find-root / find-minimum / find-maximum) names WHAT the search
looks for; the `algorithm` argument names HOW it searches. The engines (TODO 33) fall
into two families.

The MINIMISERS drive the folded objective (fold_objective(), 32.1) to its least, so
every objective works on any of them: golden-section search (31.7) and Brent's parabolic
minimiser (33.12) shrink a 1-D bracket, Nelder-Mead (33.14) walks a simplex over n
unknowns — multivariate, bounds-clamped to each bracket.

The BRACKETERS are find-root only: rather than minimising |expr| they hunt a SIGN CHANGE
of the raw signed expression and shrink the straddling interval. Bisection (33.1) halves
it, Ridders (33.5) takes an exponential-fit step, Brent-Dekker (33.2) interpolates
inverse-quadratically with a halving fallback, Chandrupatla (33.7) admits that same
interpolation under a sharper criterion, and Secant (33.3) chases the chord through the
last two points with a bisection safeguard. All five share ONE harness — `bracketed_root`,
which owns the scan, the evaluation, the polish and every error path — so each engine is
only its refinement step (`_refine_*`, registered in BRACKETED_ROOT_ENGINES). See 33.25.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from mcp_abacus.expr.nodes import EvalError, Node
from mcp_abacus.expr.value import Mode, UndefinedVariableError, Value, VariableStore
from mcp_abacus.suggest import did_you_mean


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
    falls back to a golden-section step; ``BISECTION`` is the single-variable root
    finder (33.1) that brackets a sign change and halves it — robust, but find-root
    only (an extremum has no sign change to straddle); ``RIDDERS`` is its superlinear
    sibling (33.5), the same bracket but an exponential-fit step instead of the
    midpoint; ``BRENT_DEKKER`` (33.2) is the third of that family, interpolating
    inverse-quadratically with a bisection fallback — Brent's ROOT method, distinct
    from the ``BRENT_PARABOLIC`` minimiser above; ``CHANDRUPATLA`` (33.7) is the fourth,
    the same interpolation admitted by a sharper test that keeps it off multiple roots;
    ``SECANT`` (33.3) is the plainest of the five, stepping to where the chord through the
    last two points crosses zero, with a bisection safeguard whenever that leaves the
    bracket; ``NELDER_MEAD`` is the multivariate downhill simplex (33.14). The enum value
    is the string reported in the reply's ``algorithm`` field (32.3).
    """

    GOLDEN_SECTION = "golden-section-search"  # one unknown, shrink a bracket
    BRENT_PARABOLIC = "brent-parabolic"  # one unknown, parabola + golden fallback
    BISECTION = "bisection"  # one unknown, halve a sign-changing bracket (root only)
    RIDDERS = "ridders"  # one unknown, exponential-fit on a sign-changing bracket (root only)
    BRENT_DEKKER = "brent-dekker"  # one unknown, interpolate + bisect a sign change (root only)
    CHANDRUPATLA = "chandrupatla"  # one unknown, interpolate under a sharper test (root only)
    SECANT = "secant"  # one unknown, chord through the last two points (root only)
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
# Note the deliberate split of the two Brents: bare ``brent`` keeps resolving to the
# PARABOLIC MINIMISER it has always named (33.12), so no existing call changes meaning,
# while the root finder (33.2) canonically spells out ``brent-dekker`` and takes the
# ``brent-root`` / ``brent-method`` / ``dekker`` spellings a caller is likely to reach for.
_ALGORITHM_ALIASES: dict[str, Algorithm] = {
    "golden-section": Algorithm.GOLDEN_SECTION,
    "golden": Algorithm.GOLDEN_SECTION,
    "brent": Algorithm.BRENT_PARABOLIC,
    "parabolic": Algorithm.BRENT_PARABOLIC,
    "bisect": Algorithm.BISECTION,
    "binary-search": Algorithm.BISECTION,
    "ridder": Algorithm.RIDDERS,
    "ridders-method": Algorithm.RIDDERS,
    "brent-root": Algorithm.BRENT_DEKKER,
    "brent-method": Algorithm.BRENT_DEKKER,
    "brent-dekker-method": Algorithm.BRENT_DEKKER,
    "dekker": Algorithm.BRENT_DEKKER,
    "zbrent": Algorithm.BRENT_DEKKER,
    "chandrupatlas": Algorithm.CHANDRUPATLA,
    "chandrupatla-method": Algorithm.CHANDRUPATLA,
    "secant-method": Algorithm.SECANT,
    "chord": Algorithm.SECANT,
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
        # 43.5: offer the nearest valid engine (over canonical names + aliases).
        hint = did_you_mean(algorithm, [a.value for a in Algorithm] + list(_ALGORITHM_ALIASES))
        raise SolverError(
            f"Unknown algorithm: {algorithm!r}.{hint} Valid algorithms: {valid}."
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


def autodetect_variable(node: Node) -> str:
    """Infer the sole free variable to solve for, or raise SolverError (43.3).

    When the single-unknown form omits `variable`, the unknown is the one name the
    program REFERENCES but does not ASSIGN — a program like `12*n - (450 + 3*n)`
    has exactly one such name (`n`); an assigned name (`r = 0.05`) is a computed
    constant, not free. Detection is refused unless that free name is unique: zero
    free names (nothing to solve for) or more than one (the solver cannot guess
    which is the unknown) raise SolverError telling the caller to name `variable`.
    This is the same free = referenced - assigned notion `validate_unknown` checks.
    """
    free = sorted(node.referenced_names() - node.assigned_names())
    if len(free) == 1:
        return free[0]
    if not free:
        raise SolverError(
            "Cannot auto-detect the variable to solve for: the expression has no "
            "free variable. Name the unknown explicitly via 'variable'."
        )
    names = ", ".join(repr(n) for n in free)
    raise SolverError(
        f"Cannot auto-detect the variable to solve for: the expression has "
        f"{len(free)} free variables ({names}). Name the intended unknown "
        f"explicitly via 'variable'."
    )


# --- the search engines (31.7 golden-section, 33.12 Brent, 33.1 bisection, 33.14 NM) -
# The MINIMISERS fold find-root/find-maximum into a quantity whose LEAST is the answer
# (fold_objective(), 32.1), so they need not know the objective beyond that fold. They
# are derivative-free and drive the abacus engine in the active mode — each candidate is
# materialised as a mode-faithful Value, the program evaluated, and its Value reduced
# back to a float to compare. Golden-section and Brent shrink a 1-D bracket; Nelder-Mead
# walks an n-vertex simplex over n unknowns.
# The BRACKETERS — bisection (33.1), Ridders (33.5), Brent-Dekker (33.2) and Chandrupatla
# (33.7) — do NOT minimise: they bracket a SIGN CHANGE of the raw signed expression, so
# all four are find-root only. They share one harness (`bracketed_root`, 33.25) and
# differ only in their refinement step. The reply names which ran (Algorithm, 32.3) so
# the engines are distinguishable.

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
_FLOAT_SNAP_DECIMALS = 6  # float snap-polish ladder: try clean roundings to 0..6 decimals
_FLOAT_EPS = 2.0**-52  # a double's machine epsilon — Brent-Dekker's minimum-step floor
# widens with |x| by this, since a double resolves no finer than a few ULPs out there
_SCAN_CELLS = 64  # coarse grid the bracketers' initial sign-change hunt samples across


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


def _round_decimals(x: float, ndigits: int) -> float:
    """Round ``x`` to ``ndigits`` decimal places through the language's own round().

    Goes the long way round — float -> Value -> ``Value.round_`` -> float — so the snap
    polish reuses the SAME rounding the expression language exposes (``round()``, value.py)
    rather than a private re-implementation. In floating-point mode this is exactly Python's
    half-to-even ``round(x, ndigits)``, but routing through Value keeps the two in lockstep
    if the language's rounding ever changes.
    """
    nd = Value.from_real(float(ndigits), Mode.FLOATING_POINT, 0)
    return Value.from_real(x, Mode.FLOATING_POINT, 0).round_(nd).to_float()


def _float_snap_polish(
    centre: list[float],
    probe: Callable[[list[float]], float | None],
    bounds: list[tuple[float, float]],
    cap: int = _FLOAT_SNAP_DECIMALS,
) -> None:
    """Floating-point counterpart to the fixed-point / rational grid polish (31.7).

    Floating-point has no grid to polish, so a converged candidate can sit a few ULPs off a
    clean value — 4.999999999999984 for 5, 3.9999999717 for 4. Re-probe the CLEAN roundings
    of ``centre`` (to 0, 1, …, ``cap`` decimal places) and let the caller's best-across-all
    tracking adopt one only when it is STRICTLY better: a clean root gives ``|expr| = 0`` and
    a clean optimum gives the exact extremum, either strictly below the drifted residual,
    while an irrational answer's rounding is a step AWAY from the true minimiser and so is
    strictly worse and rejected. The objective itself is the discriminator — no closeness
    threshold is needed. Ascending ``k`` with strict ``<`` means the cleanest representation
    (fewest decimals) is the one kept.

    ``probe`` is the engine's ``evaluate_objective`` adapted to take a point as a list, so the
    single-variable (golden / Brent) and multivariable (Nelder-Mead) engines share one helper;
    it has the same best-tracking side effect and ``+inf``-on-domain-error behaviour. For each
    ``k`` both the WHOLE point rounded together (the clean lattice point — one shot for an
    all-clean answer, robust to a non-separable objective) and each axis rounded ALONE (so a
    mix of one clean and one irrational coordinate still cleans the clean axis) are probed.
    Every probe is clamped to ``bounds`` so a rounding never escapes the caller's box — the
    single-variable engines do not otherwise clamp. Mirrors grid polish: fixed centre, no
    compounding, side effect through ``probe`` only; the caller skips it on timeout.
    """

    def clamp(point: list[float]) -> list[float]:
        return [min(max(point[i], lo), hi) for i, (lo, hi) in enumerate(bounds)]

    for k in range(cap + 1):
        probe(clamp([_round_decimals(c, k) for c in centre]))  # the whole clean lattice point
        for i in range(len(centre)):  # then each axis on its own, others left as found
            trial = list(centre)
            trial[i] = _round_decimals(centre[i], k)
            probe(clamp(trial))


class _Probe(Protocol):
    """An engine's candidate evaluator, as the shared polish helper needs to call it.

    Every engine closes over its own ``evaluate_objective``: it materialises ``x`` as a
    mode-faithful Value, evaluates the program, and tracks the best result seen as a side
    effect. They differ in what they RETURN — the bracketers hand back the signed value
    (``None`` on a domain error) while the minimisers hand back the folded objective
    (``+inf``) — but :func:`_polish_best` only ever calls them for that side effect, so
    the widest return type covers both. ``accept_ties`` is the one keyword it needs:
    the snap polish passes it so a clean rounding that merely TIES still wins.
    """

    def __call__(self, x: float, *, accept_ties: bool = False) -> float | None: ...


def _polish_best(
    best_solution: Value | None,
    mode: Mode,
    scale: int,
    timed_out: bool,
    probe: _Probe,
    bounds: tuple[float, float],
) -> None:
    """Re-probe around the best point so a clean answer is not missed by a hair (33.25).

    The tail every SINGLE-unknown engine runs after convergence, shared by the four
    bracketers (through :func:`bracketed_root`) and by :func:`search` /
    :func:`brent_parabolic`. Two mutually exclusive polishes, picked by mode:

      - fixed-point / rational have a GRID, and the continuous search stops within a
        bracket narrower than one grid step, so the best candidate may sit one step off an
        EXACTLY representable root. Re-test the grid neighbours (±1, ±2 steps) so a root
        on the grid is found exactly (residual 0) rather than rejected as a hair too large.
      - floating-point has no grid, so a converged candidate can sit a few ULPs off a
        clean value (4.999999999999984 for 5). :func:`_float_snap_polish` re-probes the
        clean roundings and lets best-tracking adopt one only when it is no worse.

    Both work purely through ``probe``'s best-tracking side effect — nothing is returned.
    Skipped when nothing evaluated, and on timeout: the hard cap is already spent, so
    there is no budget for extra probes.

    Nelder-Mead does NOT use this: its grid polish walks each axis of an n-dimensional
    point and clamps to the box, so it keeps its own (see :func:`nelder_mead`).
    """
    if best_solution is None or timed_out:
        return
    if mode is not Mode.FLOATING_POINT:
        step = 10.0**-scale
        centre = best_solution.to_float()
        for k in (-2, -1, 1, 2):
            probe(centre + k * step)
    else:
        _float_snap_polish(
            [best_solution.to_float()],
            lambda point: probe(point[0], accept_ties=True),
            [bounds],
        )


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

    def evaluate_objective(x: float, *, accept_ties: bool = False) -> float:
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
        # The search loop keeps the strict best; snap polish passes accept_ties so a clean
        # rounding that merely TIES (a flat optimum the drifted point already reached to the
        # last ULP) still replaces it with the tidier value.
        if (obj <= best_obj) if accept_ties else (obj < best_obj):
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

    # Grid / float-snap polish around the best point (33.25) — the shared tail every
    # single-unknown engine runs, so a root that is exactly representable is not missed
    # by a quantised hair and a drifted float lands on its clean value.
    _polish_best(best_solution, mode, scale, timed_out, evaluate_objective, (lower, upper))

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

    def evaluate_objective(x: float, *, accept_ties: bool = False) -> float:
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
        # accept_ties: snap polish replaces the best on a tie with the tidier rounding; the
        # search loop (accept_ties=False) keeps its strict best. See search() for the why.
        if (obj <= best_obj) if accept_ties else (obj < best_obj):
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

    # Grid / float-snap polish around the best point (33.25), identical to
    # golden-section's — the shared tail of every single-unknown engine.
    _polish_best(best_solution, mode, scale, timed_out, evaluate_objective, (lower, upper))

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


# --- The sign-change bracketers: one harness, five refinement steps (33.7 / 33.25) ---
# Five engines find a root the same way: hunt an interval whose endpoints straddle zero
# (opposite signs => a root between them, by the intermediate value theorem) and shrink
# it while keeping the straddle. Because they need a straddle they are find-root ONLY —
# an extremum has no sign change to bracket — and because the caller's endpoints need not
# already straddle, each first SCANS the bracket on a coarse grid for the first
# sign-changing cell. If no sign change exists anywhere (e.g. an even-multiplicity root
# that only touches zero) they say so distinctly; that case is the |expr|-minimisers'
# (golden / brent-parabolic) to cover, so the two families stay complementary.
#
# ALL of that — the objective guard, the candidate evaluation and best-tracking, the
# scan, the polish, the error paths and the SolverResult — lives once, in
# :func:`bracketed_root`. An engine is therefore ONLY its refinement step: a `_refine_*`
# function that shrinks a straddling bracket, registered in BRACKETED_ROOT_ENGINES.
# Adding a bracketer is a `_refine_*` plus one registry line, with no change here and none
# in server.py. That was the point of 33.25, which collapsed four hand-rolled copies (~86
# identical lines apiece) into this; secant (33.3) was the first engine to arrive as just
# its ~30-line step.

# The candidate evaluator a refinement step is handed: returns the SIGNED expression
# value at x so the step can compare signs, or None where x raises a DOMAIN error (no
# real value to take a sign of). Tracking the best |expr| seen is its side effect.
_Evaluate = Callable[[float], float | None]

# A refinement step: shrink the straddling bracket [a, b] (values fa, fb) and report
# (iterations, timed_out). It may assume fa and fb straddle zero — or that the bracket is
# degenerate (a == b), an exact root already sitting on a scan node. It must respect both
# caps: stop after _MAX_ITERATIONS steps, and stop and flag timed_out once `deadline`
# passes, since a single pathological evaluation can outlast the iteration cap alone.
_RefineStep = Callable[[float, float, float, float, _Evaluate, float, float], tuple[int, bool]]


def _refine_bisection(
    a: float,
    b: float,
    fa: float,
    fb: float,
    evaluate: _Evaluate,
    x_tol: float,
    deadline: float,
) -> tuple[int, bool]:
    """Halve the straddling bracket until it is narrower than the mode resolves (33.1).

    The robust baseline of the family, and the simplest possible refinement: take the
    midpoint, keep whichever half still straddles zero. One bit of the answer per step —
    linear convergence, ~35 steps to close a unit bracket to 1e-12 where the interpolating
    siblings need a handful — but it cannot be misled, which is exactly why the others
    fall back to it. ``fb`` is unused: halving needs only the sign at the moving end.
    """
    iterations = 0
    timed_out = False
    while (b - a) > x_tol and iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        m = (a + b) / 2
        fm = evaluate(m)
        if fm is None:  # a domain error opened up inside the cell — stop here
            break
        if fm == 0.0:  # landed exactly on the root
            break
        if (fm < 0) == (fa < 0):  # m has a's sign — the root is in [m, b]
            a, fa = m, fm
        else:  # opposite sign — the root is in [a, m]
            b = m
        iterations += 1
    return iterations, timed_out


def _refine_ridders(
    a: float,
    b: float,
    fa: float,
    fb: float,
    evaluate: _Evaluate,
    x_tol: float,
    deadline: float,
) -> tuple[int, bool]:
    """Shrink the bracket by Ridders' exponential fit (33.5).

    Bisection's superlinear sibling: instead of halving, each step fits an EXPONENTIAL
    through the two ends and the midpoint and solves that fit for its root in closed
    form — ``xnew = xm + (xm - xl)·sign(fl - fh)·fm / sqrt(fm² - fl·fh)``, a point
    provably inside the bracket (``|fm/s| <= 1`` since ``fl·fh < 0``). So it keeps
    bisection's robustness while converging at order ~1.84, roughly √2 of the answer per
    evaluation against bisection's one bit per step.

    The bracket can become unordered (``xl > xh``) after a re-form, so the width test and
    midpoint use abs / the mean, both order-independent (the step magnitude is bounded by
    ``|xm - xl|``, so ``xnew`` always lands back inside the interval).
    """
    iterations = 0
    timed_out = False
    xl, xh, fl, fh = a, b, fa, fb
    while abs(xh - xl) > x_tol and iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        xm = 0.5 * (xl + xh)
        fm = evaluate(xm)
        if fm is None or fm == 0.0:  # domain error inside, or the midpoint is the root
            break
        s = math.sqrt(fm * fm - fl * fh)
        if s == 0.0:  # degenerate fit — no Ridders step to take
            break
        xnew = xm + (xm - xl) * ((1.0 if fl >= fh else -1.0) * fm / s)
        fnew = evaluate(xnew)
        if fnew is None or fnew == 0.0:  # domain error, or landed exactly on the root
            break
        iterations += 1
        # Re-form the bracket around xnew from whichever pair still straddles zero.
        if (fm < 0) != (fnew < 0):  # midpoint and new estimate straddle
            xl, fl, xh, fh = xm, fm, xnew, fnew
        elif (fl < 0) != (fnew < 0):  # lower end and new estimate straddle
            xh, fh = xnew, fnew
        else:  # upper end and new estimate straddle
            xl, fl = xnew, fnew
    return iterations, timed_out


def _refine_brent_dekker(
    a: float,
    b: float,
    fa: float,
    fb: float,
    evaluate: _Evaluate,
    x_tol: float,
    deadline: float,
) -> tuple[int, bool]:
    """Shrink the bracket by inverse quadratic interpolation, else bisect (33.2).

    Brent's ROOT method — the bracketed default of most numerical libraries (Numerical
    Recipes ``zbrent``, SciPy ``brentq``), and not to be confused with the similarly named
    :func:`brent_parabolic`, which MINIMISES. Each step proposes an inverse quadratic
    interpolation through the three latest points — or Dekker's secant when only two are
    distinct and the quadratic is degenerate — and accepts it only if the move stays well
    inside the bracket AND is under half the step before last; otherwise it bisects. That
    guard is what makes the method both superlinear on smooth roots and never much worse
    than bisection on awkward ones.

    The textbook loop: ``xb`` is the running estimate, ``xc`` the contrapoint on the far
    side of the root (so f(xb)·f(xc) <= 0 throughout), ``xa`` the PREVIOUS estimate that
    the interpolation reads as its third point. ``d`` is the step just taken and ``e`` the
    one before it — the interpolation is trusted only while it keeps moving less than
    ``e``, which is what forces a bisection whenever it stops making progress.

    The other formulation of this method (Wikipedia's, with the explicit mflag) guards the
    same way but keys its stall test on the raw tolerance, which stalls into near-pure
    bisection once the estimate has converged and only the far end of the bracket is left
    to close — measurably slower here (28 iterations against 5 on exp(x) - 5), hence this
    form. On a MULTIPLE root, where interpolation gains nothing, this engine runs at about
    three times bisection's step count; :func:`_refine_chandrupatla` is the one to reach
    for there.
    """
    iterations = 0
    timed_out = False
    xa, xb, fxa, fxb = a, b, fa, fb
    xc, fxc = xb, fxb
    d = e = xb - xa
    while iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        if (fxb > 0) == (fxc > 0):  # xc fell to xb's side — re-take xa as contrapoint
            xc, fxc = xa, fxa
            d = e = xb - xa
        if abs(fxc) < abs(fxb):  # the contrapoint is the better estimate — rotate
            xa, xb, xc = xb, xc, xb
            fxa, fxb, fxc = fxb, fxc, fxb
        # The smallest step worth taking: the mode's tolerance, widened near large |x|
        # where a double resolves no finer than a few ULPs anyway.
        tol1 = 2 * _FLOAT_EPS * abs(xb) + 0.5 * x_tol
        xm = 0.5 * (xc - xb)  # a bisection step would move this far
        if abs(xm) <= tol1 or fxb == 0.0:  # bracket closed, or exactly on the root
            break
        if abs(e) >= tol1 and abs(fxa) > abs(fxb):
            s = fxb / fxa
            if xa == xc:  # only two distinct points — Dekker's secant
                p, q = 2 * xm * s, 1 - s
            else:  # three distinct points — inverse quadratic through them
                q, r = fxa / fxc, fxb / fxc
                p = s * (2 * xm * q * (q - r) - (xb - xa) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            # Accept only a step that stays well inside the bracket AND is under half
            # the step before last; otherwise fall back to bisection.
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:  # the last step was already at the tolerance — bisect
            d = e = xm
        xa, fxa = xb, fxb  # the old estimate becomes the interpolation's third point
        # Never move less than tol1, or the search stalls short of the bracket.
        xb += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fs = evaluate(xb)
        if fs is None:  # a domain error opened up inside the bracket — stop here
            break
        fxb = fs
        iterations += 1
    return iterations, timed_out


def _refine_chandrupatla(
    a: float,
    b: float,
    fa: float,
    fb: float,
    evaluate: _Evaluate,
    x_tol: float,
    deadline: float,
) -> tuple[int, bool]:
    """Shrink the bracket by interpolation admitted under Chandrupatla's test (33.7).

    Brent-Dekker's direct rival, and the sharpest of the family on awkward roots. It
    proposes the SAME inverse quadratic interpolation, but where Brent guards the step
    with a chain of heuristics, Chandrupatla (1997) asks ONE geometric question about the
    three points themselves — writing ``xi = (a-b)/(c-b)`` and ``phi = (fa-fb)/(fc-fb)``,
    the interpolation is taken exactly when

        1 - sqrt(1 - xi) < phi < sqrt(xi)

    which is precisely the region where the fitted curve is monotone across the bracket
    and so cannot propose a point outside it. Otherwise it bisects.

    The pay-off is on MULTIPLE roots, where interpolation is worthless and Brent's looser
    heuristics keep accepting it anyway: here the criterion rejects it outright and the
    search runs at bisection's own rate (~35 steps on a triple root, against Brent's
    ~105). On a simple smooth root the two are level.

    Chandrupatla's own naming: ``xa`` is the newest point, ``xb`` the far end of the
    bracket (so f(xa)·f(xb) <= 0 throughout) and ``xc`` the point before last, which the
    interpolation reads as its third. ``t`` is the NEXT sample as a fraction of the way
    from xa to xb — 0.5 being a plain bisection, which is how it starts.
    """
    iterations = 0
    timed_out = False
    xb, fxb = a, fa
    xa, fxa = b, fb
    xc, fxc = xa, fxa
    t = 0.5
    while iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        if xa == xb:  # a degenerate bracket (an exact root on a scan node) — done
            break
        xt = xa + t * (xb - xa)
        ft = evaluate(xt)
        if ft is None:  # a domain error opened up inside the bracket — stop here
            break
        iterations += 1
        # Re-form the bracket around the new point, keeping the straddle: if xt landed
        # on xa's side it simply replaces xa, otherwise the old xa becomes the far end.
        if (ft < 0) == (fxa < 0):
            xc, fxc = xa, fxa
        else:
            xc, fxc = xb, fxb
            xb, fxb = xa, fxa
        xa, fxa = xt, ft
        # Converged once the bracket is narrower than twice the tolerance (tl > 0.5),
        # measured against whichever end currently has the smaller |f|.
        xm, fm = (xa, fxa) if abs(fxa) < abs(fxb) else (xb, fxb)
        if xa == xb:
            break
        tl = (2 * _FLOAT_EPS * abs(xm) + x_tol) / abs(xb - xa)
        if tl > 0.5 or fm == 0.0:
            break
        # Chandrupatla's criterion: take the inverse quadratic interpolation only in
        # the region where the fitted curve is monotone across the bracket (so the
        # proposed point provably lands inside it); otherwise bisect. The distinctness
        # guards keep the divisions defined — a repeated abscissa or value degenerates
        # the fit, which is itself a reason to bisect.
        t = 0.5
        if xc != xb and fxc != fxb and fxa != fxb and fxa != fxc:
            xi = (xa - xb) / (xc - xb)
            phi = (fxa - fxb) / (fxc - fxb)
            if 0.0 < xi < 1.0 and 1 - math.sqrt(1 - xi) < phi < math.sqrt(xi):
                t = (fxa / (fxb - fxa)) * (fxc / (fxb - fxc)) + ((xc - xa) / (xb - xa)) * (
                    fxa / (fxc - fxa)
                ) * (fxb / (fxc - fxb))
        # Keep the next sample at least tl in from either end, so it never lands ON a
        # bracket endpoint and stalls.
        t = min(max(t, tl), 1 - tl)
    return iterations, timed_out


def _refine_secant(
    a: float,
    b: float,
    fa: float,
    fb: float,
    evaluate: _Evaluate,
    x_tol: float,
    deadline: float,
) -> tuple[int, bool]:
    """Step to where the chord through the last two points crosses zero (33.3).

    The plainest superlinear root finder there is, and the one the fancier members of this
    family fall back on: draw the straight line through the two latest iterates and take
    its zero, ``x2 = x1 - f1·(x1 - x0)/(f1 - f0)``. That is Newton's step with the
    derivative replaced by a finite difference over the points already paid for, so it
    needs NO derivative and costs ONE evaluation per step — converging at order φ ≈ 1.618,
    slower than Newton's 2 but at half the work per step.

    Textbook secant keeps the last two iterates whatever their signs, which is what makes
    it fast and also what lets it wander off — a near-flat chord throws the next point far
    outside the interval, and the iteration can diverge or cycle. Here the harness has
    already handed over a straddling cell, so that failure is simply fenced off: ``lo`` /
    ``hi`` track the sign change alongside the iteration, and any step landing outside them
    (or a degenerate ``f1 == f0`` chord with no zero to take) is replaced by a bisection of
    the safeguard bracket. The iteration itself is untouched — the safeguard only fires
    where plain secant would have failed, so the ~1.618 order stands on well-behaved roots.

    Note what this is NOT: false position / regula falsi, which keeps whichever OLD point
    preserves the straddle and thereby stalls to linear convergence on a convex curve. This
    discards the older point unconditionally, and the bracket exists only as a fence.

    On a SIMPLE root this is the leanest engine of the five — 4 steps on ``x**2 - 2``,
    3 on ``sin(x)`` near pi, where Brent-Dekker needs 4 and bisection 35. The weakness is
    the textbook one: on a REPEATED root, where f and f' vanish together, the chord's
    slope collapses with the function and the order drops to linear (rate 1 - 1/m for
    multiplicity m) — 81 steps on the triple root of ``x**3`` against Chandrupatla's 35,
    and the ``_MAX_ITERATIONS`` cap on ``x**15``. Chandrupatla is the one to reach for
    there; it detects exactly that case and bisects instead.

    Two termination tests, and both are needed. The bracket width covers the safeguarded
    path (each bisection halves it), but on the fast path the far end of the bracket may
    never move at all — secant converging from one side pins the root to full precision
    with ``hi`` still where the scan left it. So the chord's own step ``|x2 - x1|`` ends
    the search once it drops below what the mode resolves, and it is tested BEFORE the
    safeguard: the converged iterate has by then become an endpoint, so the chord aims at
    that endpoint, fails the strictly-interior test, and would otherwise hand a solved
    problem to ~20 bisections of the leftover interval.
    """
    iterations = 0
    timed_out = False
    lo, hi, flo = a, b, fa  # the safeguard bracket: straddles throughout
    x0, f0 = a, fa
    x1, f1 = b, fb
    while iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        if f1 == 0.0 or abs(hi - lo) <= x_tol:  # exactly on the root, or bracket closed
            break
        # The smallest step worth taking: the mode's tolerance, widened near large |x|
        # where a double resolves no finer than a few ULPs anyway.
        tol1 = 2 * _FLOAT_EPS * abs(x1) + 0.5 * x_tol
        denom = f1 - f0
        # inf for a flat chord: no zero to step to, so the safeguard below takes over.
        x2 = x1 - f1 * (x1 - x0) / denom if denom != 0.0 else math.inf
        if abs(x2 - x1) <= tol1:  # the chord no longer moves — converged, x1 IS the root
            break
        if not lo < x2 < hi:  # the chord aimed outside the straddle (or had no zero)
            x2 = 0.5 * (lo + hi)  # safeguard: bisect instead
        f2 = evaluate(x2)
        if f2 is None:  # a domain error opened up inside the bracket — stop here
            break
        iterations += 1
        x0, f0 = x1, f1
        x1, f1 = x2, f2
        # Keep the safeguard straddling: x2 replaces whichever end shares its sign.
        if (f2 < 0) == (flo < 0):
            lo, flo = x2, f2
        else:
            hi = x2
    return iterations, timed_out


# Every sign-change engine this build has, by the Algorithm the caller names (32.3).
# Membership doubles as the "is this a bracketed root finder?" test the server dispatches
# on, so registering an engine here is all it takes to expose it.
BRACKETED_ROOT_ENGINES: dict[Algorithm, _RefineStep] = {
    Algorithm.BISECTION: _refine_bisection,
    Algorithm.RIDDERS: _refine_ridders,
    Algorithm.BRENT_DEKKER: _refine_brent_dekker,
    Algorithm.CHANDRUPATLA: _refine_chandrupatla,
    Algorithm.SECANT: _refine_secant,
}


def bracketed_root(
    node: Node,
    variable: str,
    lower: float,
    upper: float,
    mode: Mode,
    floor: int,
    objective: Objective,
    algorithm: Algorithm,
) -> SolverResult:
    """Find a root of the program by bracketing a sign change over ``[lower, upper]``.

    The shared harness behind bisection (33.1), Ridders (33.5), Brent-Dekker (33.2),
    Chandrupatla (33.7) and secant (33.3): ``algorithm`` selects the refinement step from
    ``BRACKETED_ROOT_ENGINES`` and is echoed in the result, everything around it is here
    (33.25). The counterpart to :func:`search` / :func:`brent_parabolic`: instead of
    minimising the folded ``|expr|`` it works on the RAW signed expression, hunting an
    interval whose endpoints straddle zero — opposite signs mean a root between them, by
    the intermediate value theorem. It is therefore find-root ONLY, and raises SolverError
    for any other objective.

    The caller's endpoints need NOT already straddle zero: a coarse scan across
    ``[lower, upper]`` (``_SCAN_CELLS`` cells) locates the FIRST sign-changing cell and
    the refinement runs on that, so a root inside a same-sign bracket is still found. A
    scan point that raises a DOMAIN error yields no signed value and its cells are skipped
    (a structural failure — a constant the program never set — still propagates as an
    EvalError). The 2-second wall-clock and iteration caps bound the search exactly as in
    :func:`search`, and the same grid / float-snap polish (:func:`_polish_best`) lands a
    grid-representable root exactly.

    Returns the best candidate as a SolverResult. Raises SolverError when the program
    evaluates nowhere in the bracket, when NO sign change exists anywhere in it (a
    distinct message: widen / move the bracket, or use a minimiser for a touch-root), or
    when the straddled root is not representable on the mode's grid (the same no-solution
    message :func:`search` reports).
    """
    if objective is not Objective.FIND_ROOT:
        raise SolverError(
            f"The {algorithm.value} algorithm only finds roots (it brackets a sign "
            f"change), so objective {objective.value!r} is not supported. Use "
            f"golden-section or brent-parabolic with that objective for an extremum."
        )
    refine = BRACKETED_ROOT_ENGINES[algorithm]
    scale = _search_scale(node, mode, floor)
    x_tol, residual_tol = _tolerances(mode, scale)
    deadline = time.monotonic() + _TIME_LIMIT_SECONDS
    best_obj = math.inf
    best_solution: Value | None = None
    best_value: Value | None = None

    def evaluate_objective(x: float, *, accept_ties: bool = False) -> float | None:
        # Returns the SIGNED expression value at x (so the refinement can compare signs),
        # or None when x raises a DOMAIN error (no real value to take a sign of). Tracks
        # the best |expr| seen as a side effect — the same best-across-all bookkeeping as
        # the minimising engines, so the polish and the final result reuse it.
        nonlocal best_obj, best_solution, best_value
        candidate = Value.from_real(x, mode, scale)
        store = VariableStore()
        store.set(variable, candidate)
        try:
            raw = node.evaluate(mode, floor, variables=store)
        except EvalError as exc:
            if isinstance(exc.__cause__, UndefinedVariableError):
                raise  # a constant the program never set — structural, surface it
            return None  # a domain error at THIS candidate — no signed value here
        signed = raw.to_float()
        obj = abs(signed)  # = fold_objective(raw, FIND_ROOT).to_float(), the |expr| fold
        if (obj <= best_obj) if accept_ties else (obj < best_obj):
            best_obj, best_solution, best_value = obj, candidate, raw
        return signed

    # Coarse scan for the first sign-changing cell. A cell with a domain-error endpoint is
    # skipped (no sign to compare); an exact zero sitting on a scan node is a root already,
    # taken as a degenerate [node, node] bracket the refinement then recognises. Both
    # endpoint values are kept — the interpolating engines need the pair, and bisection
    # simply ignores the upper one.
    timed_out = False
    width = upper - lower
    xs = [lower + width * k / _SCAN_CELLS for k in range(_SCAN_CELLS + 1)]
    fxs: list[float | None] = []
    for x in xs:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        fxs.append(evaluate_objective(x))
    a: float | None = None
    b: float | None = None
    fa = fb = 0.0
    for i in range(len(fxs) - 1):
        left, right = fxs[i], fxs[i + 1]
        if left is None or right is None:
            continue
        if left == 0.0:  # an exact root sitting on a scan node — no bracket to refine
            a = b = xs[i]
            fa = fb = left
            break
        if (left < 0) != (right < 0):  # opposite signs straddle a root
            a, b, fa, fb = xs[i], xs[i + 1], left, right
            break

    iterations = 0
    if a is not None and b is not None and not timed_out:
        iterations, timed_out = refine(a, b, fa, fb, evaluate_objective, x_tol, deadline)

    # Grid / float-snap polish around the best point, so a root sitting exactly on the
    # mode's grid is found (residual 0) rather than missed by a quantised hair, and a
    # drifted float crossing at 1.999…97 lands on 2.
    _polish_best(best_solution, mode, scale, timed_out, evaluate_objective, (lower, upper))

    if best_solution is None or best_value is None:
        limit = f" within the {_TIME_LIMIT_SECONDS:g}s time limit" if timed_out else ""
        raise SolverError(
            f"The expression could not be evaluated anywhere in [{lower}, {upper}]"
            f"{limit} (every candidate for {variable!r} raised a domain error)."
        )
    if best_obj > residual_tol:
        limit = (
            f" The search stopped at the {_TIME_LIMIT_SECONDS:g}s time limit." if timed_out else ""
        )
        if a is None:  # the scan found no sign change anywhere in the bracket
            raise SolverError(
                f"No sign change for {variable!r} in [{lower}, {upper}], so "
                f"{algorithm.value} has no root to bracket. Widen or move the bracket — "
                f"or, for a root that only touches zero without crossing (even "
                f"multiplicity), use golden-section / brent-parabolic with "
                f"objective='find-minimum' (or 'find-maximum'). The closest is "
                f"|expr| = {best_obj:.6g} at {variable} = {best_solution.to_string()}.{limit}"
            )
        raise SolverError(  # a straddle was found but the root is off the mode's grid
            f"No solution: the expression does not reach zero for {variable!r} in "
            f"[{lower}, {upper}]. The closest is |expr| = {best_obj:.6g} "
            f"at {variable} = {best_solution.to_string()}.{limit}"
        )
    return SolverResult(
        variable,
        objective,
        algorithm.value,
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

    def evaluate_objective(point: list[float], *, accept_ties: bool = False) -> float:
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
        # accept_ties: snap polish replaces the best on a tie with the tidier rounding; the
        # simplex loop (accept_ties=False) keeps its strict best. See search() for the why.
        if (obj <= best_obj) if accept_ties else (obj < best_obj):
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

    # Float snap polish: the float counterpart to the grid polish above. _float_snap_polish
    # rounds each axis (and the whole point) onto its clean value and lets best-tracking adopt
    # the result when no worse — so a drifted (2.999…, 6.999…) lands on (3, 7).
    if best_point is not None and mode is Mode.FLOATING_POINT and not timed_out:
        _float_snap_polish(
            list(best_point),
            lambda point: evaluate_objective(point, accept_ties=True),
            list(zip(lowers, uppers, strict=True)),
        )

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
