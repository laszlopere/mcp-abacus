# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""FastMCP application for mcp-abacus. All tools register on this app."""

import platform
from importlib.metadata import version

from mcp.server.fastmcp import FastMCP

from mcp_abacus import __version__
from mcp_abacus.expr import parser, reference
from mcp_abacus.expr.lexer import LexError
from mcp_abacus.expr.nodes import EvalError, Node
from mcp_abacus.expr.parser import ParseError
from mcp_abacus.expr.value import FixedPoint, Mode, Value, resolve_mode

mcp = FastMCP("mcp-abacus")


@mcp.tool()
def info() -> dict:
    """Report mcp-abacus server availability, version, and environment information."""
    return {
        "status": "available",
        "name": "mcp-abacus",
        "version": __version__,
        "python": platform.python_version(),
        "mcp_sdk": version("mcp"),
        "toolsets": [],
    }


@mcp.tool(name="help")
def help_(section: str) -> str:
    """Return mcp-abacus reference text for one section, to drive the evaluator.

    Sections: 'types' (the numeric types this build supports), 'language' (the
    expression grammar — operators, precedence, literal forms), and 'functions'
    (the callable functions and their argument counts). An unknown section returns
    the list of valid section names instead of erroring.
    """
    return reference.render(section)


def _annotate(
    rendered: str,
    exact: bool,
    precision: int | None,
    offer: "tuple[Value, int] | None",
    floor_given: int | None,
) -> str:
    """Glue the exact/inexact verdict onto the rendered value (25.1, 25.3.2/3).

    The value string itself must carry the verdict so a result is never mistaken
    for exact when it was rounded: a bare `249869445103539` from `/` reads as
    exact but is not. Exact -> "(exact)"; inexact -> "(inexact)", and when the
    mode rounded at a known scale (fixed-point) that scale is named too, so an
    inexact fixed-point result reports the precision it was rounded at.

    An inexact fixed-point result ALSO steers the caller toward more fixed-point
    precision rather than toward float (25.3.2): the one case the
    min_fixed_point_precision argument helps, so the annotation names it. It names
    the argument WITHOUT a concrete value on purpose — a number there would read
    as a ceiling; the caller picks how much more. Floating-point's bare
    "(inexact)" gets no steer — the argument is invalid there and float is the
    wrong direction.

    The steer is shown ONLY when no floor is in effect for this rendering
    (``floor_given is None``). Once a floor is engaged — the caller passed
    min_fixed_point_precision, or this is the offered_precision what-if which is a
    floor by definition — the caller already knows the knob, so repeating "pass
    min_fixed_point_precision for more" is noise; the verdict shrinks to
    "(inexact, rounded to N decimals)". This mirrors the offered_precision field,
    which is likewise suppressed once a floor is given.

    When an ``offer`` (the offered_precision what-if, 25.3.3/27) was computed the
    steer carries a concrete worked example inline — "e.g. =K → 395883.8247" — so
    a caller reading only the string still sees the digits the rounding hid, not
    just the argument's name. ``offer`` is the (previewed Value, floor) pair; the
    inline hint uses the BARE rendered value, while the parallel offered_precision
    field annotates it (27.4/27.6). This keeps the string honest on its own.
    (``offer`` is non-None only when ``floor_given is None``, so it never appears
    alongside a suppressed steer.)
    """
    if exact:
        return f"{rendered} (exact)"
    if precision is not None:
        unit = "decimal" if precision == 1 else "decimals"
        rounded = f"{rendered} (inexact, rounded to {precision} {unit}"
        if floor_given is not None:
            return f"{rounded})"  # floor already engaged — no steer
        steer = f"{rounded} — pass min_fixed_point_precision for more"
        if offer is not None:
            previewed, floor = offer
            return f"{steer}; e.g. ={floor} → {previewed.to_string()})"
        return f"{steer})"
    return f"{rendered} (inexact)"


_OFFERED_BUMP = 4  # 25.3.3: an offer reveals the result scale + this many more decimals.


def _offered_precision(
    node: Node, mode: Mode, floor_given: int | None, value: Value, precision: int | None
) -> tuple[Value, int] | None:
    """A what-if at higher fixed-point precision as ``(Value, floor)``, or None (25.3.3/27.2).

    Steers an inexact fixed-point result toward more precision by SHOWING the
    digits it hid, not merely naming the argument: re-evaluates the SAME parsed
    expression with the floor raised to ``precision + _OFFERED_BUMP`` and returns
    the resulting Value alongside that floor. The caller turns the pair into the
    structured ``offered_precision`` field (annotated value + hex dump, 27.3-27.5)
    and into the inline worked example in the top-level value string (27.6). Gated
    to the one case the argument helps — fixed-point mode, an inexact result, and
    the caller did NOT already pass min_fixed_point_precision (None, not 0; an
    explicit floor means they have already engaged the knob, so no nudge). Re-
    evaluating is safe and cannot newly fail: a higher scale only pads decimals, so
    an expression that already evaluated keeps evaluating (18.5 lets the node be
    re-run).

    Returns None when a gate fails, or when the extra precision does not actually
    change the value (the "27.0000" case — every revealed digit is zero), so an
    offer never just restates the result with trailing zeros.
    """
    if mode is not Mode.FIXED_POINT or floor_given is not None or value.exact:
        return None
    assert precision is not None  # fixed-point always carries a scale
    floor = precision + _OFFERED_BUMP
    previewed = node.evaluate(mode, floor)
    assert isinstance(previewed.payload, FixedPoint) and isinstance(value.payload, FixedPoint)
    # Both stand for the same number iff the revealed lower digits are all zero:
    # compare the previewed mantissa to the result's, re-scaled up to the same floor.
    if previewed.payload.mantissa == value.payload.mantissa * 10 ** (floor - precision):
        return None
    return previewed, floor


def _resolve_mode_and_precision(
    mode: str, min_fixed_point_precision: int | None
) -> tuple[Mode | None, str | None]:
    """Resolve `mode` and validate `min_fixed_point_precision`, shared by every tool.

    Returns ``(resolved_mode, None)`` on success or ``(None, message)`` on the first
    bad argument. min_fixed_point_precision is valid ONLY in fixed-point mode (the
    other modes have no decimal scale) and must be a non-negative integer. Factored
    out of the parse/evaluate front-end so calculate, analyze, and solver share one
    mode/precision contract — the same resolution, the same checks, and the same
    error wording, whichever tool flags it.
    """
    try:
        selected = resolve_mode(mode)
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        return None, f"Unknown mode: {mode!r}. Valid modes: {valid}."
    if min_fixed_point_precision is not None:
        if selected is not Mode.FIXED_POINT:
            return None, (
                f"min_fixed_point_precision is only valid in fixed-point mode, "
                f"not {selected.value}."
            )
        if min_fixed_point_precision < 0:
            return None, (
                f"min_fixed_point_precision must be a non-negative integer, "
                f"got {min_fixed_point_precision}."
            )
    return selected, None


def _evaluate_request(
    expression: str, mode: str, min_fixed_point_precision: int | None
) -> tuple[Node | None, Mode | None, Value | None, str | None]:
    """Shared front-end for calculate/analyze: validate args, parse, evaluate.

    Resolves the mode and validates min_fixed_point_precision via
    ``_resolve_mode_and_precision``, parses the expression, and evaluates the tree in
    that mode. Returns ``(node, mode, value, None)`` on success — the evaluated root
    node, the resolved mode, and the root Value — or ``(None, None, None, message)``
    on the first failure, so each tool can wrap that message in its own reply shape.
    """
    selected, error = _resolve_mode_and_precision(mode, min_fixed_point_precision)
    if error is not None:
        return None, None, None, error
    assert selected is not None  # error is None means a mode resolved
    try:
        node = parser.parse(expression)
        value = node.evaluate(selected, min_fixed_point_precision or 0)
    except (LexError, ParseError, EvalError) as exc:
        return None, None, None, f"error (line {exc.line}): {exc.message}"
    return node, selected, value, None


@mcp.tool()
def calculate(
    expression: str, mode: str = "fixed-point", min_fixed_point_precision: int | None = None
) -> dict:
    """Evaluate an expression (or short program) in one numeric type; return value + precision.

    `mode` is the numeric type the WHOLE calculation runs in — every intermediate
    result behaves exactly as that type would, so float rounding, fixed-point
    scale, and rational exactness each show through. Modes:
      fixed-point   (default) exact scaled integer; money / ERC-20-safe
      floating-point  IEEE-754 double; ~15-17 sig. digits; aliases float64, double
      rational        exact numerator/denominator; no irrationals

    Grammar. Binary `+ - * / // %`; unary prefix `+ - ~`; `**` is POWER,
    right-assoc, binds tighter than unary minus: -2**2 == -(2**2). Bitwise
    `& | ^` (^ is XOR, NOT power) and `~` (NOT) work in EVERY type, on its own
    stored bits (float's 64-bit IEEE pattern, fixed-point's mantissa, rational's
    numerator/denominator). Both operands
    of a binary op must share ONE type — there is no implicit promotion. Group
    with `( )`. Functions: call as `name(arg, ...)` — e.g. `sqrt`, `sin`, `sum`;
    each argument evaluates in the active type. For the full set and their
    argument counts call `help('functions')`. Literals: decimals
    `12 3.14 .5 1e3 2.5e-4`; base integers
    `0x1F 0b1010 0o17`; fixed-point `M@D == M x 10^-D`, where M MUST be
    base-prefixed (0x/0o/0b) — a DECIMAL mantissa is INVALID: both `123@2` and
    `123.45@2` error; write a decimal value as its digits (e.g. 123.45), never
    with `@`. (`0x59682F00@9` = 1.5, `0xDE0B6B3A7640000@18` = 1 ETH.)

    Variables & multi-line programs. Assign with `name = expr` (name is an
    identifier `[A-Za-z_][A-Za-z0-9_]*`); a bare `name` reads it back, and reading
    a name that was never assigned is an error. An assignment is itself an
    expression — its value is the right-hand side — so `x = 2 + 3` returns 5 and
    also binds `x`. Pass SEVERAL statements as one `expression` by separating them
    with NEWLINES (`\n`): they run top to bottom sharing one variable scope, so a
    later line sees earlier bindings, and the call's returned `value` is the LAST
    statement's (earlier lines run for their bindings). E.g.
    `"x = 10\ny = x * 2\ny + 1"` returns 21. Scope lasts for the one call only —
    bindings do not carry over to the next `calculate`.

    Returns a dict: `value` is the result rendered as a string and ANNOTATED with
    its precision verdict — "(exact)" when the result is the true value, else
    "(inexact, rounded to N decimals)" — so e.g. `/` cannot silently mislead by
    looking exact when it rounded. `value_hex_dump` is that same value in hex (the
    bit-backed representation): fixed-point as M@D (mantissa in whole-byte hex,
    `@scale` dropped at scale 0), floating-point as the raw 64-bit IEEE-754
    pattern, and NULL in rational mode (a numerator/denominator pair has no single
    integer to dump). `mode` is the RESOLVED numeric type the call ran in — always
    the canonical name even when you passed an alias (e.g. "double" reads back as
    "floating-point"), so the reply stands on its own. The exactness/scale facts
    are also returned as separate fields: `exact` (bool — did the mode hold the
    true value) and `precision` (the fixed-point decimal scale, or null when the
    mode has none). NOTE: floating-point conservatively reports `exact: false` for
    every result today, including ones a double holds exactly. On failure `value`/
    `value_hex_dump`/`mode`/`exact`/`precision` are null and `error` carries the
    message (the 1-based source line for a malformed expression / domain error, or
    the valid mode list for an unknown mode); on success `error` is null. For the
    full reference call `help`.

    `min_fixed_point_precision` floors the fixed-point result at that many decimal
    places: every operand is held at no fewer than that many fractional digits, so
    a `/` that would otherwise round at scale 0 keeps more decimals. It is valid
    ONLY in fixed-point mode (the other modes have no decimal scale) and must be a
    non-negative integer; either violation is an `error`. Omit it (null) for no
    floor.

    `offered_precision` is a what-if nudge, present (non-null) ONLY on an inexact
    fixed-point result when you did NOT pass min_fixed_point_precision: it shows the
    SAME expression at a few more decimals so you see the digits the rounding hid.
    It is a nested object `{mode, min_fixed_point_precision, value, value_hex_dump,
    exact}` mirroring the top-level reply — its `mode` is always "fixed-point", its
    min_fixed_point_precision is the argument to pass to GET that fuller value, its
    `value` is what you'd get back (annotated with its OWN precision verdict, since
    the offered value may itself still be inexact — e.g. 10/3 never terminates),
    and `value_hex_dump` is that offered value in hex. It is NOT the answer to the
    call you made (that stays in the top-level `value`); it is null whenever there
    is nothing to offer.
    """
    node, selected, value, error = _evaluate_request(expression, mode, min_fixed_point_precision)
    if error is not None:
        return _error(error)
    assert node is not None and selected is not None and value is not None  # error is None
    precision = value.precision()
    offer = _offered_precision(node, selected, min_fixed_point_precision, value, precision)
    offered_precision = None
    if offer is not None:
        offered, floor = offer
        offered_precision = {
            "mode": offered.mode.value,
            "min_fixed_point_precision": floor,
            "value": _annotate(
                offered.to_string(), offered.exact, offered.precision(), None, floor
            ),
            "value_hex_dump": offered.hex_dump(),
            "exact": offered.exact,
        }
    return {
        "value": _annotate(
            value.to_string(), value.exact, precision, offer, min_fixed_point_precision
        ),
        "value_hex_dump": value.hex_dump(),
        "mode": selected.value,
        "exact": value.exact,
        "precision": precision,
        "offered_precision": offered_precision,
        "error": None,
    }


@mcp.tool()
def analyze(
    expression: str, mode: str = "fixed-point", min_fixed_point_precision: int | None = None
) -> dict:
    """Evaluate an expression and return its AST as an indented tree of sub-results.

    Same arguments and evaluation as `calculate` — `mode` (fixed-point default,
    floating-point, rational) and `min_fixed_point_precision` behave identically —
    but instead of one final value this returns the WHOLE parse tree, each node
    annotated with the Value it computed in that mode. Reach for it to see WHERE a
    surprising answer comes from: which sub-expression rounded, overflowed, or lost
    precision, rather than only the rounded result.

    `tree` is a multi-line string, one node per line, indented by depth (root last-
    applied operator at the top, literals at the leaves). Each line is
    `<OPCODE/LITERAL "lexeme"> Value = <value> (<type>[<scale>], <exact|inexact>)`
    followed by ` · `-separated per-mode details: the value in hex (fixed-point as
    M@D with whole-byte digits, `@<scale>` dropped at scale 0; float as raw IEEE-754
    bits), or a rational's decimal approximation. The `<scale>` is the fixed-point
    decimal scale (omitted for modes without one). For example `(1 + 1/2) * 3` in
    fixed-point:

        BINARY_MUL Value = 3 (fixed-point[0], inexact) · hex 0x03
          BINARY_ADD Value = 1 (fixed-point[0], inexact) · hex 0x01
            LITERAL "1" Value = 1 (fixed-point[0], exact) · hex 0x01
            BINARY_DIV Value = 0 (fixed-point[0], inexact) · hex 0x00
              LITERAL "1" Value = 1 (fixed-point[0], exact) · hex 0x01
              LITERAL "2" Value = 2 (fixed-point[0], exact) · hex 0x02
          LITERAL "3" Value = 3 (fixed-point[0], exact) · hex 0x03

    — the `1/2 = 0` leaf (inexact, scale 0) makes plain that fixed-point rounded the
    half away, so the product is 3, not 4.5, and every node above it inherits the
    inexactness. (Raise min_fixed_point_precision, or use a different mode, to keep
    those digits.)

    On success `tree` is the rendering and `error` is null; on a bad mode, an invalid
    min_fixed_point_precision, or a malformed/erroring expression, `tree` is null and
    `error` carries the message (the same messages `calculate` returns).
    """
    node, _selected, _value, error = _evaluate_request(expression, mode, min_fixed_point_precision)
    if error is not None:
        return {"tree": None, "error": error}
    assert node is not None  # error is None means the tree evaluated
    return {"tree": node.pretty(), "error": None}


@mcp.tool()
def solver(
    expression: str,
    variable: str,
    lower: float,
    upper: float,
    goal: str | None = None,
    type: str | None = None,
    mode: str = "fixed-point",
    min_fixed_point_precision: int | None = None,
) -> dict:
    """Find the value of one variable that solves or optimises an expression.

    `solver` takes the SAME expression language as `calculate` — every operator,
    function, literal form, and (crucially) multi-line programs with `name = expr`
    assignments — but instead of evaluating the expression it SEARCHES for the value
    of one named `variable` that drives the expression to a target:
      - no `goal` (default): SOLVE — find where the expression equals zero. Write an
        equation `f = g` as the expression `f - g` and solve for its root.
      - `goal` "minimise" / "maximise": OPTIMISE — find where the expression reaches
        its smallest / largest value.

    `variable` is the single unknown the search drives; it must occur in the
    expression and must NOT be assigned by it. Every OTHER name in the expression is
    a constant, set by an assignment line in the program (e.g.
    `"r = 0.05\\np = 1000\\np * (1 + r)**n - 2000"` solving for `n` with `r`, `p`
    fixed); a name that is neither the unknown nor assigned is an error.

    `lower` and `upper` give the required search bracket `[lower, upper]` the unknown
    is searched within; `lower` must be below `upper`.

    `type` (optional) names the strategy — "solve" or "optimise"; when omitted it is
    inferred from `goal` (a goal means optimise, no goal means solve).

    `mode` and `min_fixed_point_precision` behave exactly as in `calculate`: the
    search runs in that numeric type, and the found value is reported in it. See
    `calculate` and `help` for the shared grammar, modes, and precision rules.
    """
    raise NotImplementedError("solver engine is not yet implemented (TODO 31.2-31.8)")


def _error(message: str) -> dict:
    """A calculate result carrying only an error — every value field null (27.1/27.5).

    value/value_hex_dump/mode/exact/precision/offered_precision are all null: no
    mode ever resolved (an unknown mode never produced one), no value to dump, and
    nothing to offer. Same key set as the success reply so the shape never varies.
    """
    return {
        "value": None,
        "value_hex_dump": None,
        "mode": None,
        "exact": None,
        "precision": None,
        "offered_precision": None,
        "error": message,
    }


# FUTURE (SA.1): opt-in toolset gating goes here. When the feature toolsets
# (numeric | bits | eval) exist, read the enabled set from config/env at
# import time and register only those tools, mirroring mcp-tmux's opt-in
# toolset pattern. Until then every tool below registers unconditionally.
