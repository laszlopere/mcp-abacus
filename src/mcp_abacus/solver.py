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

The TANGENT engines — Newton-Raphson (33.4) and Halley (33.8) — are find-root only like
the bracketers but belong to NEITHER family: they hold no bracket at all, following the
expression's own derivatives from a single seed point. Newton steps along the tangent line
(`x - f/f'`), Halley along a tangent hyperbola that carries the curvature as well
(`x - 2ff'/(2f'^2 - ff'')`), both differenced from the same five-point stencil the
language's `diff` uses (40.17) — one set of samples, so Halley's higher order is free.
That buys a root a sign change cannot bracket (an even-multiplicity one that only touches
zero) at the price of a method that can wander, so the bracket the caller gives becomes a
fence rather than a straddle. They too share ONE harness (`tangent_root`), each engine
being only its step function (`_step_*`, registered in TANGENT_ROOT_ENGINES).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from mcp_abacus.expr.forms import finite_difference_step
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
    falls back to a golden-section step; ``TERNARY`` (33.6) is the third of that family,
    shrinking the bracket by thirds instead of by the golden ratio — the plainest
    unimodal minimiser, and the slowest of the three; ``NEWTON_OPTIMISE`` (33.13) is the
    fourth and the only one of them that is not derivative-free — it steps to the zero of
    the objective's own SLOPE, ``x - g'/g''``, and so finds an extremum rather than a root
    (the only engine in the family restricted that way); ``BISECTION`` is the single-variable root
    finder (33.1) that brackets a sign change and halves it — robust, but find-root
    only (an extremum has no sign change to straddle); ``RIDDERS`` is its superlinear
    sibling (33.5), the same bracket but an exponential-fit step instead of the
    midpoint; ``BRENT_DEKKER`` (33.2) is the third of that family, interpolating
    inverse-quadratically with a bisection fallback — Brent's ROOT method, distinct
    from the ``BRENT_PARABOLIC`` minimiser above; ``CHANDRUPATLA`` (33.7) is the fourth,
    the same interpolation admitted by a sharper test that keeps it off multiple roots;
    ``SECANT`` (33.3) is the plainest of the five, stepping to where the chord through the
    last two points crosses zero, with a bisection safeguard whenever that leaves the
    bracket; ``NEWTON_RAPHSON`` (33.4) and ``HALLEY`` (33.8) are the root finders that need
    NO sign change — they follow the expression's own derivatives to zero from a single seed
    point, Newton along the tangent LINE and Halley along a tangent hyperbola carrying the
    curvature too (cubic convergence, for the same evaluation budget); ``NELDER_MEAD`` is
    the multivariate downhill simplex (33.14). The enum value is the string reported in the
    reply's ``algorithm`` field (32.3).
    """

    GOLDEN_SECTION = "golden-section-search"  # one unknown, shrink a bracket
    BRENT_PARABOLIC = "brent-parabolic"  # one unknown, parabola + golden fallback
    TERNARY = "ternary-search"  # one unknown, shrink a bracket by thirds
    NEWTON_OPTIMISE = "newton-optimise"  # one unknown, step to the slope's zero (extrema only)
    BISECTION = "bisection"  # one unknown, halve a sign-changing bracket (root only)
    RIDDERS = "ridders"  # one unknown, exponential-fit on a sign-changing bracket (root only)
    BRENT_DEKKER = "brent-dekker"  # one unknown, interpolate + bisect a sign change (root only)
    CHANDRUPATLA = "chandrupatla"  # one unknown, interpolate under a sharper test (root only)
    SECANT = "secant"  # one unknown, chord through the last two points (root only)
    NEWTON_RAPHSON = "newton-raphson"  # one unknown, follow the tangent to zero (root only)
    HALLEY = "halley"  # one unknown, tangent hyperbola — Newton plus curvature (root only)
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
    "ternary": Algorithm.TERNARY,
    "ternary-section": Algorithm.TERNARY,
    "trisection": Algorithm.TERNARY,
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
    # Bare ``newton`` names the ROOT finder — the same first-come rule bare ``brent`` follows
    # above. 33.13's gradient minimiser duly spells itself out as ``newton-optimise``, so no
    # existing call changed meaning when it landed.
    "newton": Algorithm.NEWTON_RAPHSON,
    "newton-method": Algorithm.NEWTON_RAPHSON,
    "newtons-method": Algorithm.NEWTON_RAPHSON,
    "newton raphson": Algorithm.NEWTON_RAPHSON,
    "raphson": Algorithm.NEWTON_RAPHSON,
    # 33.13: the American spelling, and the spaced forms. Every one of them keeps the
    # ``-optimise`` half — a bare ``newton`` must stay the root finder it has always named.
    "newton-optimize": Algorithm.NEWTON_OPTIMISE,
    "newton optimise": Algorithm.NEWTON_OPTIMISE,
    "newton optimize": Algorithm.NEWTON_OPTIMISE,
    "halleys": Algorithm.HALLEY,
    "halley-method": Algorithm.HALLEY,
    "halleys-method": Algorithm.HALLEY,
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
# back to a float to compare. Golden-section, Brent-parabolic and ternary shrink a 1-D
# bracket and share one harness (`minimise`, 33.6); Nelder-Mead walks an n-vertex
# simplex over n unknowns and keeps its own (it is multivariate).
# The BRACKETERS — bisection (33.1), Ridders (33.5), Brent-Dekker (33.2), Chandrupatla
# (33.7) and secant (33.3) — do NOT minimise: they bracket a SIGN CHANGE of the raw signed
# expression, so all five are find-root only. They share one harness (`bracketed_root`,
# 33.25) and differ only in their refinement step. The TANGENT engines — Newton-Raphson
# (33.4) and Halley (33.8) — are find-root only too but hold no bracket at all: they follow
# finite-difference DERIVATIVES from a seed point, fenced into the caller's interval, and
# share a harness of their own (`tangent_root`). The reply names which ran (Algorithm, 32.3)
# so the engines are distinguishable.

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


# --- The single-variable minimisers: one harness, four search loops (33.6 / 33.13) ---
# Four engines answer the same question the same way: drive the caller's interval
# toward the LEAST of the folded objective (fold_objective(), 32.1) — ``|expr|`` for a
# find-root, ``±expr`` for an extremum — evaluating the program at points of their own
# choosing until the answer settles below what the mode can resolve. Because they order a
# folded quantity rather than compare SIGNS, they are the single-variable engines that
# serve every objective, and the ones that reach a root which merely touches zero (no sign
# change for the bracketers to straddle).
#
# Three of them SHRINK the interval and are derivative-free (golden-section, ternary,
# brent-parabolic). The fourth, newton-optimise (33.13), instead uses it as a FENCE and
# steps along the objective's own derivatives to the zero of its slope — so it is the one
# engine here that cannot serve find-root, whose ``|expr|`` fold has a kink exactly where
# it would be aiming (see _OPTIMISE_ONLY).
#
# Everything around the loop — the candidate materialisation and best-tracking, the
# scale / tolerance / deadline / step setup, the polish and the error paths and the
# SolverResult — lives once, in :func:`minimise`. An engine is therefore ONLY its loop:
# a `_loop_*` function driving ``[a, b]``, registered in MINIMISER_ENGINES. This is
# the same split 33.25 made for the bracketers (`bracketed_root`) and `tangent_root` made
# for the derivative engines, deliberately deferred at the time because the minimisers
# differ from the bracketers in the one place that matters — what the evaluator RETURNS —
# and so could not share THAT harness. Ternary (33.6) was the engine whose arrival made a
# third hand-rolled copy the alternative; it landed as just its ~20-line loop.
#
# Nelder-Mead (33.14) stays outside: it is multivariate, so its bracket, its polish and
# its result are all n-dimensional and nothing here fits it.

# The candidate evaluator a minimiser loop is handed: returns the FOLDED objective at x.
# A DOMAIN error there yields ``+inf`` — the point simply looks maximally bad, so the loop
# steers away from a region the program cannot evaluate without needing to know why.
# Tracking the best seen is its side effect. Contrast the bracketers' `_Evaluate`, which
# returns the SIGNED value and ``None``: a sign is what THEY compare, and ``+inf`` has no
# sign to offer.
_MinimiseEvaluate = Callable[[float], float]

# A loop: drive ``[a, b]`` toward the objective's least, evaluating through the callable,
# and return ``(iterations, timed_out)``. It owns its whole convergence — the ``x_tol``
# stop, the ``_MAX_ITERATIONS`` cap, and the ``deadline`` check that sets ``timed_out`` —
# and returns nothing about the best point, which reaches the harness through the
# evaluator's side effect instead.
#
# The trailing ``h`` is the mode's finite-difference step (``finite_difference_step``),
# which only a DERIVATIVE loop needs; the three bracket-shrinking loops accept and ignore
# it. It is passed rather than recomputed because deriving it needs the mode and working
# scale, which the harness resolves and a loop never sees.
_MinimiseLoop = Callable[[float, float, _MinimiseEvaluate, float, float, float], tuple[int, bool]]


def _loop_golden_section(
    a: float,
    b: float,
    evaluate: _MinimiseEvaluate,
    x_tol: float,
    deadline: float,
    h: float,  # unused: this loop is derivative-free
) -> tuple[int, bool]:
    """Shrink the bracket by the golden ratio (31.7) — the default engine.

    Hold two interior points at the golden fractions of ``[a, b]``, drop the end beyond
    whichever is worse, and repeat. The golden ratio is what makes it cost ONE new
    evaluation per step rather than two: the surviving interior point sits at exactly the
    right fraction of the new, narrower interval to serve as one of ITS two probes. So the
    interval falls to 0.618 of itself per evaluation, and on a unimodal objective the
    minimum can never escape the bracket. Contrast :func:`_loop_ternary`, which is this
    idea with the fractions chosen so that reuse does NOT happen.
    """
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc = evaluate(c)
    fd = evaluate(d)
    iterations = 0
    timed_out = False
    while (b - a) > x_tol and iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        if fc <= fd:
            b, d, fd = d, c, fc  # minimum is left of d; reuse c as the new d
            c = b - _INV_PHI * (b - a)
            fc = evaluate(c)
        else:
            a, c, fc = c, d, fd  # minimum is right of c; reuse d as the new c
            d = a + _INV_PHI * (b - a)
            fd = evaluate(d)
        iterations += 1
    if not timed_out:
        evaluate((a + b) / 2)  # the converged midpoint, folded into the best
    return iterations, timed_out


def _loop_ternary(
    a: float,
    b: float,
    evaluate: _MinimiseEvaluate,
    x_tol: float,
    deadline: float,
    h: float,  # unused: this loop is derivative-free
) -> tuple[int, bool]:
    """Shrink the bracket by thirds (33.6) — the plainest unimodal minimiser.

    Cut ``[a, b]`` at its two TRISECTION points ``m1 = a + w/3`` and ``m2 = b - w/3`` and
    discard the outer third on the worse side: if ``f(m1) <= f(m2)`` the minimum of a
    unimodal objective cannot lie beyond ``m2``, so ``b`` moves to ``m2``, and
    symmetrically otherwise. Same guarantee as golden-section — on a unimodal objective
    the minimum stays inside the bracket — reached by the most obvious possible split, and
    it is the textbook companion to the bisection of a sign change: where bisection asks a
    SIGN at one point, ternary must ask an ORDER between two, which is precisely why one
    probe will not do.

    It is the SLOWEST of the three minimisers, and knowing why is the reason to have it.
    It loses to golden-section TWICE over. Per step it shrinks the interval only to 2/3,
    where the golden split reaches 0.618; and each of its steps costs TWO evaluations,
    because neither trisection point of ``[a, b]`` is a trisection point of the survivor,
    so nothing carries over. That is ``(2/3)**0.5 ~ 0.816`` of the interval per evaluation
    against golden-section's 0.618, and closing the same bracket therefore costs about
    ``ln(0.618)/ln(0.816) ~ 2.4x`` the program evaluations (measured: 147 against 64 on a
    unit-scale float minimum). It is offered because it is the method a caller is most
    likely to reach for by name, not because it wins.
    """
    iterations = 0
    timed_out = False
    while (b - a) > x_tol and iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        third = (b - a) / 3
        m1, m2 = a + third, b - third
        if evaluate(m1) <= evaluate(m2):
            b = m2  # the minimum is left of m2 — drop the upper third
        else:
            a = m1  # the minimum is right of m1 — drop the lower third
        iterations += 1
    if not timed_out:
        evaluate((a + b) / 2)  # the converged midpoint, folded into the best
    return iterations, timed_out


def _loop_brent_parabolic(
    a: float,
    b: float,
    evaluate: _MinimiseEvaluate,
    x_tol: float,
    deadline: float,
    h: float,  # unused: this loop is derivative-free
) -> tuple[int, bool]:
    """Shrink the bracket by parabolic interpolation, golden-section on distrust (33.12).

    The fast one: instead of splitting the bracket at a fixed fraction, fit a parabola
    through the best three points seen and leap near its VERTEX, which on a smooth
    extremum is close to the answer after very few steps. The loop is the textbook bounded
    Brent (Numerical Recipes ``brent`` / SciPy ``fminbound``), kept inside ``[a, b]``: the
    parabolic step is accepted only when its vertex lands inside the current bracket and
    the move is below half the step-before-last (the ``e`` bookkeeping); otherwise a
    golden-section step (``_GOLDEN`` of the larger sub-bracket) is taken. A non-smooth
    objective — the kinked ``|expr|`` of a find-root — simply triggers that fallback more
    often, so it still converges. Same "trust the fancy step only inside its safe region"
    fence Brent-Dekker puts around its interpolation on the root side.
    """
    # x is the best point so far, w the second best, v the previous w; the parabola is
    # fitted through the three. d is the last step, e the step before it (the parabola
    # is only trusted when it asks for less than half of e). Seed all three at one
    # interior point, a golden fraction in from the lower end.
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = evaluate(x)
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
        fu = evaluate(u)
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
    return iterations, timed_out


_NEWTON_OPTIMISE_SEED_CELLS = _SCAN_CELLS  # the seed is the best of the same coarse scan
# the bracketers run — see _loop_newton_optimise for why a fenced method still scans first.


def _loop_newton_optimise(
    a: float,
    b: float,
    evaluate: _MinimiseEvaluate,
    x_tol: float,
    deadline: float,
    h: float,
) -> tuple[int, bool]:
    """Step to the zero of the objective's own SLOPE, ``x - g'/g''`` (33.13).

    Newton's method applied not to the objective but to its DERIVATIVE, which is what turns
    a root finder into an optimiser: a stationary point of ``g`` is a zero of ``g'``, so the
    tangent-line step that :func:`_step_newton` takes on ``f`` becomes ``x - g'/g''`` here.
    Equivalently it fits a PARABOLA through the local value, slope and curvature and jumps
    straight to that parabola's vertex — which is why it lands an exactly quadratic
    objective on the answer in a SINGLE step, where brent-parabolic needs several to build
    its three-point fit and golden-section needs dozens of interval halvings.

    THE DERIVATIVES. Numerical, from the same five-point central stencil of the language's
    ``diff`` (40.17) that :func:`tangent_root` uses, at the shared per-mode step ``h``
    (``finite_difference_step``) — so the two engines cannot drift apart:
    ``g'(x) ~ (g(x-2h) - g(x+2h) + 8*(g(x+h) - g(x-h))) / (12h)`` and
    ``g''(x) ~ (-g(x-2h) + 16g(x-h) - 30g(x) + 16g(x+h) - g(x+2h)) / (12h^2)``, both 4th
    order. They share their samples and the centre value is the one already in hand, so a
    step costs FOUR program evaluations. Note ``h`` is deliberately coarse (1e-5 in float,
    ~10**-(scale/5) in fixed-point) precisely so the second difference — which divides by
    ``12h**2`` — keeps its significant digits instead of dissolving into quantisation noise.

    THE SEED. Like :func:`tangent_root`, a derivative method wants a starting guess where
    the API gives an interval, so it takes the best of a coarse scan over ``[a, b]``
    (``_NEWTON_OPTIMISE_SEED_CELLS`` cells): the point of LEAST objective. That both seeds
    the iteration near the extremum — where ``g'' > 0`` and the parabola fit is trustworthy
    — and guarantees the engine never reports worse than a 64-point sampling of the
    interval, whatever the iteration then does.

    THE FENCE AND THE STOPS. ``[a, b]`` bounds rather than brackets: every step is clamped
    back into it, and a step clamped onto the point it started from ends the iteration (it
    wants out of the region the caller asked about). Three more stops, each a place where
    the parabola fit stops meaning anything: a stencil sample that leaves the expression's
    domain (``+inf`` from the harness's evaluator), a slope already exactly zero (standing
    on the stationary point), and — the one peculiar to optimisation — ``g'' <= 0``, where
    the curve is locally flat or CONCAVE. Stepping on a non-positive curvature would head
    for a maximum, or leap arbitrarily far on a vanishing denominator; since the seed is
    already the scan's best point, stopping there and keeping it is strictly safer. Every
    evaluation feeds the harness's best-tracking, so each of these stops leaves the best
    point found intact and the polish still runs.
    """
    # Seed: the coarse scan's point of least objective. The harness tracks the best VALUE
    # for the result; the loop needs the POINT to start stepping from, so it keeps its own.
    seed: float | None = None
    seed_obj = math.inf
    timed_out = False
    for k in range(_NEWTON_OPTIMISE_SEED_CELLS + 1):
        if time.monotonic() >= deadline:
            timed_out = True  # hard 2s cap reached — stop with the best seen so far
            break
        x = a + (b - a) * k / _NEWTON_OPTIMISE_SEED_CELLS
        obj = evaluate(x)
        if obj < seed_obj:
            seed_obj, seed = obj, x

    iterations = 0
    if seed is None or timed_out:
        return iterations, timed_out
    x, gx = seed, seed_obj
    while iterations < _MAX_ITERATIONS:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if not math.isfinite(gx):
            break  # a domain error where we stand — no parabola to fit here
        if x - 2 * h < a or x + 2 * h > b:
            # No room for a CENTRED stencil without sampling outside the caller's
            # interval. Clamping the samples would skew the differences and give a wrong
            # slope; taking them anyway would evaluate — and, through best-tracking, let
            # the engine RETURN — a point the caller never asked about, which is how a
            # find-minimum of `x` over [0, 5] once answered -2e-05. So stop instead: the
            # seed scan has already sampled both endpoints, so an optimum sitting on one
            # is held by best-tracking regardless.
            break
        gm2, gm1 = evaluate(x - 2 * h), evaluate(x - h)
        gp1, gp2 = evaluate(x + h), evaluate(x + 2 * h)
        if not all(map(math.isfinite, (gm2, gm1, gp1, gp2))):
            break  # the stencil left the domain — no local fit to step along
        slope = (gm2 - gp2 + 8 * (gp1 - gm1)) / (12 * h)
        curvature = (-gm2 + 16 * gm1 - 30 * gx + 16 * gp1 - gp2) / (12 * h * h)
        if slope == 0.0:
            break  # already standing on the stationary point
        if curvature <= 0.0:
            break  # flat or concave here — the step would climb, or explode
        x_next = min(max(x - slope / curvature, a), b)
        if x_next == x:
            break
        converged = abs(x_next - x) <= x_tol
        g_next = evaluate(x_next)
        iterations += 1
        # An extremum is FLAT, which bounds how well any method can locate it: near the
        # optimum g changes quadratically, so the objective stops distinguishing points
        # about a square-root-of-precision away — 1e-8 in float, where ``x_tol`` asks for
        # 1e-12. The slope driving the step is by then mostly stencil round-off, and the
        # step-size test alone would let the iteration wander around the optimum until the
        # iteration cap. So stop the moment a step fails to IMPROVE the objective: the
        # harness's best-tracking already holds the better point, making this free.
        if not g_next < gx:
            break
        x, gx = x_next, g_next
        if converged:  # the last step moved less than the mode resolves
            break
    return iterations, timed_out


# Every single-variable minimiser this build has, by the Algorithm the caller names (32.3).
# Membership doubles as the "is this a 1-D minimiser?" test the server dispatches on, so
# registering an engine here is all it takes to expose it.
MINIMISER_ENGINES: dict[Algorithm, _MinimiseLoop] = {
    Algorithm.GOLDEN_SECTION: _loop_golden_section,
    Algorithm.BRENT_PARABOLIC: _loop_brent_parabolic,
    Algorithm.TERNARY: _loop_ternary,
    Algorithm.NEWTON_OPTIMISE: _loop_newton_optimise,
}

# The minimisers that serve EXTREMA only. Every other engine in this family takes any
# objective, because the fold (32.1) turns a root hunt into just another minimisation of
# |expr|. A DERIVATIVE engine cannot follow it there: |expr| has a KINK exactly at the root
# it would be aiming for — the slope jumps sign across it and the curvature the step
# divides by is meaningless — so newton-optimise refuses find-root and names the engines
# built for it instead. The mirror image of the find-root-only guards in
# :func:`bracketed_root` and :func:`tangent_root`.
_OPTIMISE_ONLY: frozenset[Algorithm] = frozenset({Algorithm.NEWTON_OPTIMISE})


def minimise(
    node: Node,
    variable: str,
    lower: float,
    upper: float,
    mode: Mode,
    floor: int,
    objective: Objective,
    algorithm: Algorithm,
) -> SolverResult:
    """Drive the unknown to the objective's least over ``[lower, upper]`` (31.7 / 33.6).

    The shared harness behind golden-section (31.7), Brent-parabolic (33.12), ternary
    (33.6) and newton-optimise (33.13): ``algorithm`` selects the loop from
    ``MINIMISER_ENGINES`` and is echoed in the result, everything around it is here.
    Unlike :func:`bracketed_root` and :func:`tangent_root` this family works on the folded
    quantity (fold_objective(), 32.1) — ``|expr|`` for find-root, ``±expr`` for an extremum
    — whose least is the answer either way, so its derivative-free engines serve EVERY
    objective. The one exception is newton-optimise, which differentiates that fold and so
    cannot take find-root's kinked ``|expr|``: see ``_OPTIMISE_ONLY``.

    Each candidate is materialised in ``mode`` at the working scale, bound into a fresh
    seeded store, and the program evaluated; the resulting Value is reduced to a float to
    drive the loop. A candidate that raises a DOMAIN error (e.g. sqrt of a negative) is
    penalised with ``+inf`` so the loop steers away, but a STRUCTURAL failure — a constant
    the program never sets — propagates as an EvalError (it fails at every point, and is
    the user's to fix, not a region to avoid). The best objective across ALL evaluations is
    tracked, not just the loop's final point, so quantisation making the very last
    candidate a hair worse cannot cost the answer.

    The search is bounded by ``_MAX_ITERATIONS`` and by a hard wall-clock limit of
    ``_TIME_LIMIT_SECONDS`` (2s): the iteration cap bounds the NUMBER of evaluations, but a
    single pathological candidate can be slow, so the loop checks the deadline each step
    and stops with the best candidate reached so far. The same grid / float-snap polish
    (:func:`_polish_best`) the root engines run then lands an exactly representable answer.

    Returns the best candidate found as a SolverResult. Raises SolverError when the
    expression evaluates nowhere in the bracket, or when a find-root cannot drive |expr|
    within ``residual_tol`` of zero (reporting the closest it reached) — including when the
    time limit cut the search short before it could.
    """
    if algorithm in _OPTIMISE_ONLY and objective is Objective.FIND_ROOT:
        raise SolverError(
            f"The {algorithm.value} algorithm only finds extrema (it steps to the zero of "
            f"the objective's own slope), so objective 'find-root' is not supported: "
            f"|expr| has a kink at the root, where the curvature the step divides by does "
            f"not exist. Use newton-raphson or halley to follow derivatives to a root, or "
            f"golden-section-search / brent-parabolic / ternary-search to minimise |expr|."
        )
    loop = MINIMISER_ENGINES[algorithm]
    scale = _search_scale(node, mode, floor)
    x_tol, residual_tol = _tolerances(mode, scale)
    deadline = time.monotonic() + _TIME_LIMIT_SECONDS
    # The stencil's step, in the active mode's own grid — only the derivative loop uses it,
    # but the harness owns it because deriving it needs the mode and working scale.
    h = finite_difference_step(mode, scale).to_float()
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

    iterations, timed_out = loop(lower, upper, evaluate_objective, x_tol, deadline, h)

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
        algorithm.value,
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


# --- The tangent engines: one harness, two derivative steps (33.4 / 33.8) -----
# The third engine shape, and the only root finders here that hold no bracket: from a
# single point they take a local polynomial fit of the curve and step to where THAT crosses
# zero. Newton-Raphson (33.4) fits the tangent line, x <- x - f/f'; Halley (33.8) fits a
# hyperbola through the same point using the curvature as well, x <- x - 2ff'/(2f'^2 - ff'').
#
# The derivatives are NOT symbolic — nothing in this build differentiates an AST. They come
# from the five-point central stencil the language's own `diff` (40.17) uses, at the same
# per-mode step h (`finite_difference_step`), so the two cannot drift apart: the stencil is
# 4th-order (exact through quartics) and h tracks the active mode's working precision, which
# is what keeps a fixed-point difference from cancelling itself away. The SAME five samples
# (the point, x+-h, x+-2h) yield both f' and f'', so Halley's extra derivative is free —
# cubic convergence for exactly the evaluation budget Newton's quadratic costs.
#
# Two consequences worth naming. A step costs FIVE program evaluations (the point plus the
# four stencil samples) where bisection costs one, so "fewer iterations" is not automatically
# "less work" — it wins when the convergence order outruns that factor, which on a smooth
# simple root it does comfortably. And no sign change is required, so these reach an
# even-multiplicity root that the whole bracketed family refuses (converging linearly there,
# as every derivative method does on a repeated root).
#
# Everything but the step — the objective guard, the evaluation and best-tracking, the seed
# scan, the derivative stencil, the fence, the polish, the error paths and the SolverResult —
# lives once, in :func:`tangent_root`. An engine is therefore ONLY its step function,
# registered in TANGENT_ROOT_ENGINES; the same split 33.25 made for the bracketers.
_TANGENT_SEED_CELLS = _SCAN_CELLS  # the seed is the best of the same coarse scan the
# bracketers run — see tangent_root's docstring for why an unbracketed method still
# starts from a scan of the caller's interval.

# A step: given the expression value and its first two derivatives at the current point,
# return the DISTANCE to move back toward the root (the iteration takes x - step). It may
# assume ``d1`` is non-zero — the harness stops on a flat tangent before ever calling it —
# so a step function never divides by zero and never fails.
_TangentStep = Callable[[float, float, float], float]


def _step_newton(fx: float, d1: float, d2: float) -> float:
    """Newton's step: the tangent line's own zero, ``f / f'`` (33.4).

    Fit the curve at the current point by its TANGENT and go to where that line crosses
    zero. The error is squared each step (quadratic convergence) on a smooth simple root, so
    the correct digits double. ``d2`` is unused: a straight line has no curvature to carry —
    that is exactly what :func:`_step_halley` adds.
    """
    return fx / d1


def _step_halley(fx: float, d1: float, d2: float) -> float:
    """Halley's step: the tangent HYPERBOLA's zero, ``2ff' / (2f'^2 - ff'')`` (33.8).

    One order up from Newton for the same five evaluations. Newton's line ignores that the
    curve bends away from it; Halley fits a hyperbola matching the value, the slope AND the
    curvature, which lands nearer the root — the error is CUBED each step rather than
    squared. Equivalently it is Newton's step scaled by ``1 / (1 - f·f''/(2f'^2))``, the
    curvature correction; where the curve is locally straight (``f'' = 0``) the factor is 1
    and the two engines coincide exactly.

    The correction is trusted only while it stays a correction. ``2f'^2 - f·f''`` starts at
    ``2f'^2`` and the curvature term can shrink it to zero, or drive it NEGATIVE — far from
    the root, or when a low-precision fixed-point grid leaves the differenced ``f''`` mostly
    noise. A vanishing denominator explodes the step; a negative one reverses it, sending
    the iteration away from the root. So the step is taken only while the denominator keeps
    at least HALF of the ``2f'^2`` it started from (``>= f'^2``, a signed test that rules out
    both), which bounds the Halley step at twice Newton's and never flips its direction;
    outside that region it falls back to Newton's step. Same "trust the fancy step only
    inside its safe region" fence Brent-Dekker puts around its interpolation — it is what
    keeps Halley from failing where Newton walks in, as on a root pinned against a log's
    asymptote. Halley therefore never does much worse than Newton, and the harness's
    flat-tangent stop (``f' = 0``) remains the only way either engine runs out of step.
    """
    denominator = 2.0 * d1 * d1 - fx * d2
    if denominator < d1 * d1:  # the curvature term has eaten half the denominator, or
        return fx / d1  # turned it over — either way it is no longer a correction
    return 2.0 * fx * d1 / denominator


# Every derivative-driven engine this build has, by the Algorithm the caller names (32.3).
# Membership doubles as the "is this a tangent root finder?" test the server dispatches on,
# so registering an engine here is all it takes to expose it.
TANGENT_ROOT_ENGINES: dict[Algorithm, _TangentStep] = {
    Algorithm.NEWTON_RAPHSON: _step_newton,
    Algorithm.HALLEY: _step_halley,
}


def tangent_root(
    node: Node,
    variable: str,
    lower: float,
    upper: float,
    mode: Mode,
    floor: int,
    objective: Objective,
    algorithm: Algorithm,
) -> SolverResult:
    """Find a root by stepping along the program's own derivatives (33.4 / 33.8).

    The shared harness behind Newton-Raphson (33.4) and Halley (33.8): ``algorithm`` selects
    the step from ``TANGENT_ROOT_ENGINES`` and is echoed in the result, everything else is
    here. Find-root only (a derivative step locates a zero, not an extremum) but, unlike
    :func:`bracketed_root`'s five engines, it brackets NOTHING: it iterates from one seed
    point, converging quadratically (Newton) or cubically (Halley) on a smooth simple root —
    a handful of steps where bisection needs ~35.

    THE DERIVATIVES. Numerical, from the five-point central stencil of the language's
    ``diff`` (40.17) at the shared per-mode step ``h`` (``finite_difference_step``):
    ``f'(x) ~ (f(x-2h) - f(x+2h) + 8*(f(x+h) - f(x-h))) / (12h)`` and
    ``f''(x) ~ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)) / (12h^2)``, both 4th
    order. They share their samples, so a step costs five program evaluations whichever
    engine is running — Halley's higher order is free.

    THE SEED. A derivative method wants a starting guess; this tool's API gives a bracket.
    The engine takes the best of a coarse scan over ``[lower, upper]``
    (``_TANGENT_SEED_CELLS`` cells) — the point of least ``|expr|`` — which is the same scan
    the bracketers pay for, spent on a different question (they want the first SIGN CHANGE,
    this wants the closest approach). That makes the seed as good as the interval allows and
    keeps the engine honest on a bracket whose endpoints say nothing useful. ``iterations``
    counts steps only, as with every other engine.

    THE FENCE. Textbook Newton can leap anywhere — a near-flat tangent throws the next point
    far away, and the iteration may diverge or cycle. Here ``[lower, upper]`` is a fence
    rather than a straddle: every step is clamped back into it, and a step clamped onto the
    point it started from ends the iteration (it is trying to leave the region the caller
    asked about). A tangent that goes exactly flat, or a stencil sample that leaves the
    expression's domain, ends it too — with the best point seen so far intact, since every
    evaluation feeds the same best-tracking the other engines use.

    WHAT IT BUYS. No sign change is required, so these are the root engines that reach an
    even-multiplicity root — ``(x - pi)**2``, which only touches zero — where the whole
    bracketed family reports "no sign change". Convergence there is linear rather than
    quadratic/cubic (f and f' vanish together), the classic behaviour of a derivative method
    on a repeated root.

    The 2-second wall-clock and ``_MAX_ITERATIONS`` caps bound it exactly as elsewhere, and
    the same grid / float-snap polish (:func:`_polish_best`) lands a representable root
    exactly. Raises SolverError for a non-root objective, when the program evaluates nowhere
    in the bracket, or when the iteration does not reach zero (reporting the closest |expr|,
    and naming a flat tangent when that is what stopped it).
    """
    if objective is not Objective.FIND_ROOT:
        raise SolverError(
            f"The {algorithm.value} algorithm only finds roots (it steps along the "
            f"expression's own derivatives to where it crosses zero), so objective "
            f"{objective.value!r} is not supported. Use golden-section or brent-parabolic "
            f"with that objective for an extremum."
        )
    step_from = TANGENT_ROOT_ENGINES[algorithm]
    scale = _search_scale(node, mode, floor)
    x_tol, residual_tol = _tolerances(mode, scale)
    deadline = time.monotonic() + _TIME_LIMIT_SECONDS
    # The stencil's step, in the active mode's own grid — h must be representable there or
    # x+h would quantise straight back to x and the difference would be identically zero.
    h = finite_difference_step(mode, scale).to_float()
    best_obj = math.inf
    best_solution: Value | None = None
    best_value: Value | None = None

    def evaluate_objective(x: float, *, accept_ties: bool = False) -> float | None:
        # The signed expression value at x — the step needs the sign and magnitude both, and
        # the stencil differences these — or None where x raises a DOMAIN error. Tracks the
        # best |expr| seen as a side effect, exactly as in :func:`bracketed_root`.
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
        # Record the best only for points INSIDE the caller's bracket. The derivative
        # stencil probes x +- h and x +- 2h, which near an endpoint fall OUTSIDE [lower,
        # upper]; being real evaluations they would otherwise feed best-tracking, and one
        # landing on a root just past the endpoint would be RETURNED as the answer — how a
        # find-root of `x + 0.00001` over [0, 5] once reported x = -1e-5. Gating the record
        # (not the evaluation) keeps every probe available to the derivative while pinning
        # the reported root to the interval; an in-bracket root within 2h of an endpoint is
        # still reached, since the iteration point itself stays fenced. Contrast
        # newton-optimise (33.13), which refuses to PROBE outside its fence at all — a
        # minimiser can stop at an endpoint the seed scan already sampled, but a root finder
        # wants to keep stepping toward a near-endpoint crossing.
        in_bracket = lower <= x <= upper
        if in_bracket and ((obj <= best_obj) if accept_ties else (obj < best_obj)):
            best_obj, best_solution, best_value = obj, candidate, raw
        return signed

    def derivatives(x: float, fx: float) -> tuple[float, float] | None:
        # diff's stencil (40.17) in float, from four mode-faithful samples plus the value
        # the loop already holds — BOTH derivatives out of the same five points:
        #   f'  ~ (f(x-2h) - f(x+2h) + 8*(f(x+h) - f(x-h))) / (12h)
        #   f'' ~ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)) / (12h^2)
        # so Halley's curvature costs no evaluation Newton does not already spend. None when
        # any sample leaves the expression's domain — no local fit to step along from there.
        fm2 = evaluate_objective(x - 2 * h)
        fm1 = evaluate_objective(x - h)
        fp1 = evaluate_objective(x + h)
        fp2 = evaluate_objective(x + 2 * h)
        if fm2 is None or fm1 is None or fp1 is None or fp2 is None:
            return None
        first = (fm2 - fp2 + 8 * (fp1 - fm1)) / (12 * h)
        second = (-fm2 + 16 * fm1 - 30 * fx + 16 * fp1 - fp2) / (12 * h * h)
        return first, second

    # Seed: the coarse scan's point of least |expr|, which best-tracking already records.
    timed_out = False
    width = upper - lower
    for k in range(_TANGENT_SEED_CELLS + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        evaluate_objective(lower + width * k / _TANGENT_SEED_CELLS)

    iterations = 0
    went_flat = False  # the tangent had no slope to follow — a distinct way to stop
    if best_solution is not None and not timed_out:
        x = best_solution.to_float()
        fx = evaluate_objective(x)
        while iterations < _MAX_ITERATIONS:
            if time.monotonic() >= deadline:
                timed_out = True  # hard 2s cap reached — stop with the best seen so far
                break
            if fx is None:
                break  # a domain error where we stand — nowhere to take a tangent from
            if fx == 0.0:
                break  # an exact root: the step would be zero and every further one too
            local_fit = derivatives(x, fx)
            if local_fit is None:
                break  # the stencil left the domain — no local fit here either
            first, second = local_fit
            if first == 0.0:
                went_flat = True  # a horizontal tangent never meets zero
                break
            # The engine's step, fenced into the caller's interval. Clamping onto the point
            # we started from means the iteration wants OUT of it — stop rather than spin.
            x_next = min(max(x - step_from(fx, first, second), lower), upper)
            if x_next == x:
                break
            converged = abs(x_next - x) <= x_tol
            x = x_next
            fx = evaluate_objective(x)
            iterations += 1
            if converged:  # the last step moved less than the mode resolves
                break

    # Grid / float-snap polish around the best point (33.25) — the shared single-unknown
    # tail, so a root exactly on the mode's grid is found rather than missed by a hair.
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
        flat = (
            " The tangent went flat (zero slope) before a root was reached, so the "
            "iteration had nowhere to step; try a bracket around the crossing itself, or "
            "a sign-change engine (bisection / brent-dekker)."
            if went_flat
            else ""
        )
        raise SolverError(
            f"No solution: the expression does not reach zero for {variable!r} in "
            f"[{lower}, {upper}]. The closest is |expr| = {best_obj:.6g} "
            f"at {variable} = {best_solution.to_string()}.{limit}{flat}"
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
