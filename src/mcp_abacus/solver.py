# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The solver tool's engine: pick a strategy, then search for the variable's value.

Built up item by item (TODO 31). Today it resolves the solver STRATEGY from the
`type`/`goal` arguments; the objective and the search engine land in later items.
"""

from enum import Enum

from mcp_abacus.expr.nodes import Node
from mcp_abacus.expr.value import Value


class SolverError(Exception):
    """A solver request that cannot be honoured — a bad argument or no solution.

    Unlike LexError / ParseError / EvalError it carries no source line: a solver
    failure is about the request (its type, goal, or bracket) or the search outcome,
    not a position in the expression text.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SolverType(Enum):
    """The search strategy — drive the expression to zero, or to an extremum."""

    SOLVE = "solve"  # find where the expression equals zero (no goal)
    OPTIMISE = "optimise"  # find where the expression is smallest / largest (a goal)


class Goal(Enum):
    """The optimise direction — make the expression as small or as large as possible."""

    MINIMISE = "minimise"
    MAXIMISE = "maximise"


# Friendly spellings accepted alongside the canonical British names (31.5): the
# short forms and the American -ize endings all resolve to the same Goal.
_GOAL_ALIASES: dict[str, Goal] = {
    "min": Goal.MINIMISE,
    "minimize": Goal.MINIMISE,
    "max": Goal.MAXIMISE,
    "maximize": Goal.MAXIMISE,
}


def resolve_type(type_: str | None, goal: str | None) -> SolverType:
    """Resolve the solver strategy from `type` and `goal`, or raise SolverError (31.2).

    When `type` is omitted it is inferred from `goal`: a goal means OPTIMISE (there is
    an extremum to seek), no goal means SOLVE (drive the expression to zero). When
    `type` is given it must name a known strategy AND agree with the goal — OPTIMISE
    needs a goal to optimise toward, while SOLVE takes none (its target is fixed at
    zero). Goal presence is all that matters here; the goal's own value (minimise /
    maximise, and its aliases) is resolved with the objective in a later item.
    """
    has_goal = goal is not None
    if type_ is None:
        return SolverType.OPTIMISE if has_goal else SolverType.SOLVE
    try:
        resolved = SolverType(type_)
    except ValueError:
        valid = ", ".join(t.value for t in SolverType)
        raise SolverError(f"Unknown solver type: {type_!r}. Valid types: {valid}.") from None
    if resolved is SolverType.OPTIMISE and not has_goal:
        raise SolverError("Solver type 'optimise' requires a goal (minimise or maximise).")
    if resolved is SolverType.SOLVE and has_goal:
        raise SolverError(
            "Solver type 'solve' does not take a goal "
            "(omit goal to solve, or set type to 'optimise')."
        )
    return resolved


def resolve_goal(goal: str | None) -> Goal | None:
    """Resolve the optimise direction, or None for a solve, or raise SolverError (31.5).

    None stays None — a solve has no direction, it drives the expression to zero. A
    given goal must name a direction: the canonical ``minimise`` / ``maximise``, the
    American ``minimize`` / ``maximize``, or the short ``min`` / ``max``. Anything
    else is a SolverError listing the canonical names. (Whether a goal is allowed at
    all for the chosen strategy is settled earlier, in ``resolve_type``.)
    """
    if goal is None:
        return None
    try:
        return Goal(goal)
    except ValueError:
        if goal in _GOAL_ALIASES:
            return _GOAL_ALIASES[goal]
        valid = ", ".join(g.value for g in Goal)
        raise SolverError(f"Unknown goal: {goal!r}. Valid goals: {valid}.") from None


def objective(value: Value, goal: Goal | None) -> Value:
    """Transform the expression's Value into the quantity the search drives to its least (31.5).

    The solver always MINIMISES this quantity, so the goal is folded into the value:
      - solve (``goal`` is None) -> ``|expr|``; its least is zero, i.e. a root.
      - minimise                 -> ``expr`` unchanged.
      - maximise                 -> ``-expr``; the least of the negation is the
                                    greatest of the value.
    The transform stays in the active mode (``abs_`` / ``neg``), so the search
    compares candidates in the mode's own representation, not a float shadow.
    """
    if goal is None:
        return value.abs_()
    if goal is Goal.MAXIMISE:
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
