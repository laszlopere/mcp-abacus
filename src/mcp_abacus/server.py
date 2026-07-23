# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""FastMCP application for mcp-abacus. All tools register on this app."""

import platform
from collections.abc import Sequence as AbcSequence
from importlib.metadata import version
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock
from pydantic import Field, ValidationError

from mcp_abacus import __version__
from mcp_abacus.errors import format_validation_error
from mcp_abacus.expr import parser, reference
from mcp_abacus.expr.lexer import LexError
from mcp_abacus.expr.nodes import Assign, EvalError, Node, Sequence
from mcp_abacus.expr.parser import ParseError
from mcp_abacus.expr.value import (
    MODE_ALIASES,
    FixedPoint,
    InexactHandling,
    Mode,
    Value,
    resolve_inexact_handling,
    resolve_mode,
    selectable_modes,
)
from mcp_abacus.fit import FitError, FitResult, fit_all
from mcp_abacus.solver import (
    Algorithm,
    SolverError,
    SolverResult,
    autodetect_variable,
    bisection,
    brent_dekker,
    brent_parabolic,
    chandrupatla,
    nelder_mead,
    resolve_algorithm,
    resolve_objective,
    ridders,
    search,
    validate_bracket,
    validate_unknown,
)
from mcp_abacus.suggest import did_you_mean


class _AbacusFastMCP(FastMCP):
    """FastMCP that reshapes argument-validation errors for the model (TODO 43.2).

    FastMCP wraps a failed pydantic argument validation as a ToolError whose
    `__cause__` is the ValidationError. We catch that one case and re-raise with
    a concise, field-naming message (errors.format_validation_error); the SDK
    still returns it as an `isError` result. Tool-body failures (their own
    Lex/Parse/Eval/Solver errors) are untouched -- those carry a different cause
    or none, so they fall through unchanged.
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> AbcSequence[ContentBlock] | dict[str, Any]:
        try:
            return await super().call_tool(name, arguments)
        except ToolError as exc:
            if isinstance(exc.__cause__, ValidationError):
                raise ToolError(format_validation_error(name, exc.__cause__)) from exc.__cause__
            raise


mcp = _AbacusFastMCP(
    "mcp-abacus",
    instructions=(
        "A calculator for language models: type-faithful arithmetic you can trust "
        "and reason about. Pick a numeric type/mode and the WHOLE expression "
        "behaves exactly as that type would in real code -- it rounds where the "
        "type rounds and stays exact where the type is exact. Modes: fixed-point "
        "(default; exact scaled integer, money / ERC-20-safe), floating-point "
        "(IEEE-754 double; aliases float64, double), complex (a+bi over two "
        "fixed-point parts; imaginary unit 1i), and rational (exact "
        "numerator/denominator). Every answer is labelled with its own precision "
        "verdict (exact vs inexact, rounded to N decimals), so a result that "
        "merely looks precise can never be mistaken for the true value. Tools: "
        "`calculate` evaluates an expression; `analyze` returns the parse tree "
        "with each node's value, showing WHERE an answer rounded or overflowed; "
        "`solver` finds the variable value(s) driving an expression to a root or "
        "extremum; `curve_fit` fits curve forms to paired (x, y) data and reports each "
        "fitted equation with its error; `help` serves the grammar/type reference; "
        "`info` reports version and environment. Offline and deterministic."
    ),
)


@mcp.tool()
def info() -> dict:
    """Report mcp-abacus server availability, version, and environment information.

    `toolsets` lists the opt-in tool groups active in this build. It is EMPTY today —
    every tool registers unconditionally — and is reserved for the future toolset
    gating (SA.1); it is reported now so a client can read the active set once that
    lands.
    """
    return {
        "status": "available",
        "name": "mcp-abacus",
        "version": __version__,
        "python": platform.python_version(),
        "mcp_sdk": version("mcp"),
        "toolsets": [],
    }


@mcp.resource(
    "abacus://reference",
    name="abacus-reference-index",
    title="abacus reference index",
    description=(
        "Index of the available reference sections; read abacus://reference/"
        "{section} for one section's full text."
    ),
    mime_type="text/markdown",
)
def reference_index() -> str:
    """List the reference sections an LLM can read, one per line."""
    return reference.index()


@mcp.resource(
    "abacus://reference/{section}",
    name="abacus-reference-section",
    title="abacus reference section",
    description=(
        "The reference text for one section: 'types', 'language', 'functions', "
        "'solver', or 'fit' — the same content the `help` tool returns, exposed as a resource."
    ),
    mime_type="text/markdown",
)
def reference_section(section: str) -> str:
    """Return one reference section's text (mirrors the `help` tool)."""
    return reference.render(section)


# The valid `help` sections, advertised to clients as a schema enum. Kept in
# lockstep with reference._SECTIONS by a test; reference.render() still handles an
# unknown section gracefully for any direct caller that bypasses schema validation.
HelpSection = Literal["types", "language", "functions", "solver", "fit"]


@mcp.tool(name="help")
def help_(
    section: Annotated[
        HelpSection,
        Field(
            description=(
                "Which reference section to return: 'types', 'language', 'functions', "
                "'solver', or 'fit'."
            )
        ),
    ],
    search_filter: Annotated[
        str | None,
        Field(
            description=(
                "Optional case-insensitive substring; keeps only the section's lines that "
                "contain it (e.g. 'sin' over 'functions' returns the sin/asin/asinh/sinh rows). "
                "Omit or leave empty to return the whole section."
            )
        ),
    ] = None,
    details: Annotated[
        bool,
        Field(
            description=(
                "When true, expands the 'functions' section: each matching function is shown "
                "as a card with its signature, arity, and description instead of a one-line "
                "row. Combine with 'search_filter' to detail one function. No effect on the "
                "other sections."
            )
        ),
    ] = False,
) -> str:
    """Return mcp-abacus reference text for one section, to drive the evaluator.

    Sections: 'types' (the numeric types this build supports), 'language' (the
    expression grammar — operators, precedence, literal forms), 'functions' (the
    callable functions and their argument counts), 'solver' (the solver tool —
    solving / optimising one variable over a bracket), and 'fit' (the curve_fit tool
    — fitting curve forms to paired (x, y) data). `section` is restricted to these
    five names — advertised as a schema enum — so any other value is rejected with the
    valid list.

    `search_filter`, when given, narrows the section to the lines that contain it as
    a case-insensitive substring; a filter that matches nothing says so. `details`
    expands the 'functions' section into per-function cards (signature, arity,
    description).
    """
    return reference.render(section, search_filter, details)


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


def _offered_precision_for(
    root: Node,
    statement: Node,
    mode: Mode,
    floor_given: int | None,
    value: Value,
    precision: int | None,
) -> tuple[Value, int] | None:
    """A what-if at higher fixed-point precision for ONE statement, or None (25.3.3/27.2).

    Steers an inexact fixed-point result toward more precision by SHOWING the
    digits it hid, not merely naming the argument: re-evaluates the SAME parsed
    program (``root``) with the floor raised to ``precision + _OFFERED_BUMP`` and
    reads ``statement``'s freshly cached Value off the re-run. Re-running the WHOLE
    program — not the statement in isolation — is deliberate: a later line may read a
    name an earlier assignment bound, and ``root.evaluate`` rebuilds the whole
    VariableStore (30.2), so every binding is back in place for ``statement``. The
    caller turns the returned pair into the structured ``offered_precision`` field
    (annotated value + hex dump, 27.3-27.5) and into the inline worked example in the
    value string (27.6). Gated to the one case the argument helps — fixed-point mode,
    an inexact result, and the caller did NOT already pass min_fixed_point_precision
    (None, not 0; an explicit floor means they have already engaged the knob, so no
    nudge). Re-evaluating is safe and cannot newly fail: a higher scale only pads
    decimals, so an expression that already evaluated keeps evaluating (18.5 lets the
    node be re-run). It runs under the default CONTINUE_AND_REPORT, so a higher floor
    never trips abort-on-inexact.

    NOTE: ``root.evaluate`` OVERWRITES every node's cached ``.value`` (18.5). The
    caller must snapshot all base statement Values BEFORE invoking this for any
    statement, since this mutates the tree.

    Returns None when a gate fails, or when the extra precision does not actually
    change the value (the "27.0000" case — every revealed digit is zero), so an
    offer never just restates the result with trailing zeros.
    """
    if mode is not Mode.FIXED_POINT or floor_given is not None or value.exact:
        return None
    if precision is None:
        # The result is not a single fixed-point scalar — e.g. a VECTOR built inside
        # fixed-point mode (19.1.10) carries no scale of its own. There is no single
        # scale to bump, so make no offer (the inexact verdict still surfaces).
        return None
    floor = precision + _OFFERED_BUMP
    root.evaluate(mode, floor)
    previewed = statement.value
    assert previewed is not None  # the re-run walks every statement, caching its Value
    assert isinstance(previewed.payload, FixedPoint) and isinstance(value.payload, FixedPoint)
    # Both stand for the same number iff the revealed lower digits are all zero:
    # compare the previewed mantissa to the result's, re-scaled up to the same floor.
    if previewed.payload.mantissa == value.payload.mantissa * 10 ** (floor - precision):
        return None
    return previewed, floor


def _offered_precision(
    node: Node, mode: Mode, floor_given: int | None, value: Value, precision: int | None
) -> tuple[Value, int] | None:
    """Single-node offer: re-evaluate ``node`` itself for its own what-if (25.3.3/27.2).

    The whole-tree case of ``_offered_precision_for`` where the statement IS the root —
    kept as a named wrapper for callers that offer one bare node.
    """
    return _offered_precision_for(node, node, mode, floor_given, value, precision)


def _offer_dict(offer: "tuple[Value, int] | None") -> dict | None:
    """Render an ``(offered Value, floor)`` pair as the structured offered_precision field.

    None when there is nothing to offer; otherwise the nested object mirroring the
    top-level reply (27.3-27.5): its mode is always fixed-point, min_fixed_point_precision
    is the floor to PASS to get this fuller value, and value/value_hex_dump/exact describe
    it (annotated with its OWN verdict, since the offered value may itself still be inexact).
    """
    if offer is None:
        return None
    offered, floor = offer
    return {
        "mode": offered.mode.value,
        "min_fixed_point_precision": floor,
        "value": _annotate(offered.to_string(), offered.exact, offered.precision(), None, floor),
        "value_hex_dump": offered.hex_dump(),
        "exact": offered.exact,
    }


def _statement_result(
    value: Value, offer: "tuple[Value, int] | None", floor_given: int | None
) -> dict:
    """The common per-result fields for one Value: the calculate reply shape minus mode/values.

    The single source of truth for the annotate/hex/precision rendering, reused for both the
    top-level reply (the program's last result) and each ``values`` entry. Returns
    value/value_hex_dump/exact/precision/offered_precision; the caller adds ``source`` for a
    list entry, or ``mode``/``values``/``error`` for the top-level reply.
    """
    precision = value.precision()
    return {
        "value": _annotate(value.to_string(), value.exact, precision, offer, floor_given),
        "value_hex_dump": value.hex_dump(),
        "exact": value.exact,
        "precision": precision,
        "offered_precision": _offer_dict(offer),
    }


def _resolve_mode_and_precision(
    mode: str, min_fixed_point_precision: int | None
) -> tuple[Mode | None, str | None]:
    """Resolve `mode` and validate `min_fixed_point_precision`, shared by every tool.

    Returns ``(resolved_mode, None)`` on success or ``(None, message)`` on the first
    bad argument. min_fixed_point_precision is valid in the modes that carry a decimal
    scale — fixed-point and complex (whose two parts are fixed-point) — and must be a
    non-negative integer; the float/rational modes have no scale to floor. Factored
    out of the parse/evaluate front-end so calculate, analyze, and solver share one
    mode/precision contract — the same resolution, the same checks, and the same
    error wording, whichever tool flags it.
    """
    try:
        selected = resolve_mode(mode)
    except ValueError:
        # selectable_modes drops the internal VECTOR container (19.1.10), so a bad mode
        # is never offered or suggested "vector" — it is not a mode a caller can pick.
        names = [m.value for m in selectable_modes()]
        valid = ", ".join(names)
        # 43.5: a typo'd mode gets the nearest valid spelling (over names + aliases).
        hint = did_you_mean(mode, names + list(MODE_ALIASES))
        return None, f"Unknown mode: {mode!r}.{hint} Valid modes: {valid}."
    if min_fixed_point_precision is not None:
        if selected not in (Mode.FIXED_POINT, Mode.COMPLEX):
            return None, (
                f"min_fixed_point_precision is only valid in fixed-point and complex "
                f"modes, not {selected.value}."
            )
        if min_fixed_point_precision < 0:
            return None, (
                f"min_fixed_point_precision must be a non-negative integer, "
                f"got {min_fixed_point_precision}."
            )
    return selected, None


def _resolve_inexact_handling(name: str) -> tuple[InexactHandling | None, str | None]:
    """Resolve the calculate `inexact_handling` argument (35.2), shared error shape.

    Returns ``(handling, None)`` on success or ``(None, message)`` on an unknown
    name, listing the valid choices — the same self-correcting shape an unknown mode
    gets. Only ``calculate`` takes this argument today; analyze/solver always run the
    default CONTINUE_AND_REPORT (they are diagnostics that must not abort).
    """
    try:
        return resolve_inexact_handling(name), None
    except ValueError:
        valid = ", ".join(h.value for h in InexactHandling)
        return None, f"Unknown inexact_handling: {name!r}. Valid values: {valid}."


def _evaluate_request(
    expression: str,
    mode: str,
    min_fixed_point_precision: int | None,
    inexact_handling: InexactHandling = InexactHandling.CONTINUE_AND_REPORT,
) -> tuple[Node | None, Mode | None, Value | None, str | None]:
    """Shared front-end for calculate/analyze: validate args, parse, evaluate.

    Resolves the mode and validates min_fixed_point_precision via
    ``_resolve_mode_and_precision``, parses the expression, and evaluates the tree in
    that mode. Returns ``(node, mode, value, None)`` on success — the evaluated root
    node, the resolved mode, and the root Value — or ``(None, None, None, message)``
    on the first failure, so each tool can wrap that message in its own reply shape.

    ``inexact_handling`` (35.2) rides into ``node.evaluate``; under ABORT_ON_INEXACT
    an inexact result raises an EvalError caught here like any other evaluation
    failure, so its diagnostic flows back through the ``error`` channel. Defaults to
    CONTINUE_AND_REPORT, so analyze (which omits it) is unaffected.
    """
    selected, error = _resolve_mode_and_precision(mode, min_fixed_point_precision)
    if error is not None:
        return None, None, None, error
    assert selected is not None  # error is None means a mode resolved
    try:
        node = parser.parse(expression)
        value = node.evaluate(
            selected, min_fixed_point_precision or 0, inexact_handling=inexact_handling
        )
    except (LexError, ParseError, EvalError) as exc:
        # The message stands on its own (no machine-style "error (line N):" prefix);
        # the source line is named in prose only where it matters — the inexact-abort
        # headline (35.3.1) — not bolted onto every diagnostic.
        return None, None, None, exc.message
    return node, selected, value, None


@mcp.tool()
def calculate(
    expression: Annotated[
        str,
        Field(
            description=(
                "The expression, or a newline-separated multi-line program "
                "(`name = expr` assignments sharing one scope), to evaluate."
            )
        ),
    ],
    mode: Annotated[
        str,
        Field(
            description=(
                "Numeric type the WHOLE calculation runs in: 'fixed-point' "
                "(default; exact scaled integer, money/ERC-20-safe; alias 'decimal'), "
                "'floating-point' (IEEE-754 double; aliases 'float64', 'double', "
                "'float', 'ieee754'), 'rational' (exact numerator/denominator; "
                "aliases 'fraction', 'frac'), or 'complex' (a+bi over two fixed-point "
                "parts; write the imaginary unit as '1i', e.g. '3+4i'). Note 'decimal' "
                "resolves to fixed-point, NOT a decimal float. `help('types')` lists the full set."
            )
        ),
    ] = "fixed-point",
    min_fixed_point_precision: Annotated[
        int | None,
        Field(
            description=(
                "Floor fixed-point results at this many decimal places "
                "(non-negative integer). Valid ONLY in fixed-point mode; "
                "null for no floor."
            )
        ),
    ] = None,
    inexact_handling: Annotated[
        str,
        Field(
            description=(
                "What to do when a result is inexact: 'continue-and-report' "
                "(default; evaluate and let the precision verdict surface) or "
                "'abort-on-inexact' (fail on the first inexact sub-result)."
            )
        ),
    ] = "continue-and-report",
) -> dict:
    """Evaluate an expression (or short program) in one numeric type; return value + precision.

    Use `calculate` when you want the VALUE of an expression. To instead see WHERE a
    surprising answer rounded or overflowed — the per-node parse tree with each
    sub-result — use `analyze`; to find the variable value(s) that drive an
    expression to a root or extremum, use `solver`. All three share this expression
    language and `mode`/`min_fixed_point_precision` arguments.

    `mode` is the numeric type the WHOLE calculation runs in — every intermediate
    result behaves exactly as that type would, so float rounding, fixed-point
    scale, and rational exactness each show through. Modes:
      fixed-point   (default) exact scaled integer; money / ERC-20-safe; alias decimal
      floating-point  IEEE-754 double; ~15-17 sig. digits; aliases float64, double, float, ieee754
      rational        exact numerator/denominator; no irrationals; aliases fraction, frac
      complex         a+bi over two fixed-point parts; imaginary unit `1i` (e.g. 3+4i);
                      exact + - *, rounds / and transcendentals; no ordering/bitwise/solver

    Grammar. Binary `+ - * / // %`; unary prefix `+ - ~`; `**` is POWER,
    right-assoc, binds tighter than unary minus: -2**2 == -(2**2). Bitwise
    `& | ^` (^ is XOR, NOT power) and `~` (NOT) work in EVERY type, on its own
    stored bits (float's 64-bit IEEE pattern, fixed-point's mantissa, rational's
    numerator/denominator). Both operands
    of a binary op must share ONE type — there is no implicit promotion. Group
    with `( )`. Functions: call as `name(arg, ...)` — e.g. `sqrt`, `sin`, `sum`;
    each argument evaluates in the active type. For the full set and their
    argument counts call `help('functions')`. Constants: `pi` and `e` are usable
    bare (no parentheses), e.g. `2*pi`; assigning to them is an error. Literals: decimals
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
    later line sees earlier bindings. EVERY bare-expression line is answered, in source
    order, in the `values` array (below); assignment lines run silently for their
    bindings and are NOT echoed — except the final line, which is always the program's
    result and so is always echoed. When more than one line is answered, the top-level
    `value` is a multi-line transcript, one `<expression> = <result>` per answered line;
    a single answered line keeps `value` as just that result. E.g.
    `"x = 10\ny = x * 2\ny + 1"` returns 21 (only `y + 1` is a bare line), while
    `"x = 10\n(x - 1) / (x + 1)\n(x + 1) / 100.0"` answers both divisions. Scope lasts
    for the one call only — bindings do not carry over to the next `calculate`.

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
    every result today, including ones a double holds exactly. `values` is the per-line
    breakdown: an array with one object per answered line, in source order, each
    `{source, value, value_hex_dump, exact, precision, offered_precision}` — `source` is
    the re-rendered expression and the other fields mirror the top-level ones for that one
    line. The LAST entry is the program's result, so its fields equal the top-level
    `value_hex_dump`/`exact`/`precision`/`offered_precision` (and, for a single answered
    line, the top-level `value`). On failure `value`/`value_hex_dump`/`mode`/`exact`/
    `precision`/`offered_precision`/`values` are null and `error` carries a plain,
    self-contained message — what went wrong (a malformed expression, a domain error,
    or an unknown mode with the valid list). It reads as prose, not a log line; only
    the inexact-abort diagnostic names its source line, and in prose. On success
    `error` is null. For the full reference call `help`.

    `min_fixed_point_precision` floors the fixed-point result at that many decimal
    places: every operand is held at no fewer than that many fractional digits, so
    a `/` that would otherwise round at scale 0 keeps more decimals. It is valid
    ONLY in fixed-point mode (the other modes have no decimal scale) and must be a
    non-negative integer; either violation is an `error`. Omit it (null) for no
    floor.

    `inexact_handling` chooses what happens when a result is INEXACT:
      continue-and-report  (default) evaluate normally and let the verdict surface
                           in `value`/`exact`; never reject. Aliases: continue, report.
      abort-on-inexact     stop and FAIL the moment any sub-result is inexact. The
                           call returns no value — `error` instead carries a
                           diagnostic naming the source line and laying out the
                           operation that went inexact in computed VALUES (e.g.
                           `1.00 / 3.00 = 0.33`), then how to enable inexact
                           calculations if you do want the rounded answer. Use it
                           when an approximate answer is unacceptable and you want to
                           be told precisely what and where, rather than silently
                           trusting a rounded figure. Aliases: abort, strict, exact-only.
    An unknown value is an `error` listing the valid choices. Note floating-point
    reports every result inexact, so abort-on-inexact there fails on the first value.

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
    is nothing to offer. Each `values` entry carries its OWN `offered_precision` under
    the same gate, so every answered line steers independently.
    """
    handling, handling_error = _resolve_inexact_handling(inexact_handling)
    if handling_error is not None:
        return _error(handling_error)
    assert handling is not None  # handling_error is None means a handling resolved
    node, selected, value, error = _evaluate_request(
        expression, mode, min_fixed_point_precision, handling
    )
    if error is not None:
        return _error(error)
    assert node is not None and selected is not None and value is not None  # error is None
    # The OUTPUT statements (30.5): bare expressions echo, assignments run silently — except
    # the final statement, which is always the program's result and so is always echoed.
    statements = node.statements if isinstance(node, Sequence) else (node,)
    outputs = [s for s in statements if not isinstance(s, Assign) or s is statements[-1]]
    # Snapshot every output's BASE Value before computing any offer: each offer re-runs the
    # whole program at a higher floor, which OVERWRITES every node's cached .value (18.5).
    base: list[tuple[Node, Value]] = []
    for s in outputs:
        sval = s.value
        assert sval is not None  # the eval cached each statement's Value (18.5)
        base.append((s, sval))
    offers = [
        _offered_precision_for(node, s, selected, min_fixed_point_precision, sval, sval.precision())
        for s, sval in base
    ]
    values = [
        {"source": s.source(), **_statement_result(sval, offer, min_fixed_point_precision)}
        for (s, sval), offer in zip(base, offers, strict=True)
    ]
    # Top-level scalars stay pinned to the program's result (the last output) for backward
    # compatibility; value becomes a `source = result` transcript only when 2+ outputs exist.
    _, last_value = base[-1]
    reply = {
        **_statement_result(last_value, offers[-1], min_fixed_point_precision),
        "mode": selected.value,
        "values": values,
        "error": None,
    }
    if len(values) >= 2:
        reply["value"] = "\n".join(f"{v['source']} = {v['value']}" for v in values)
    return reply


@mcp.tool()
def analyze(
    expression: Annotated[
        str,
        Field(
            description=(
                "The expression or newline-separated program to parse and "
                "evaluate; same grammar as `calculate`."
            )
        ),
    ],
    mode: Annotated[
        str,
        Field(
            description=(
                "Numeric type to evaluate in: 'fixed-point' (default), "
                "'floating-point', or 'rational' — as in `calculate`."
            )
        ),
    ] = "fixed-point",
    min_fixed_point_precision: Annotated[
        int | None,
        Field(
            description=(
                "Floor on fixed-point fractional digits (non-negative integer); "
                "fixed-point mode only, null for no floor — as in `calculate`."
            )
        ),
    ] = None,
) -> dict:
    """Evaluate an expression and return its AST as an indented tree of sub-results.

    Same arguments and evaluation as `calculate` — `mode` (fixed-point default,
    floating-point, rational) and `min_fixed_point_precision` behave identically —
    but instead of one final value this returns the WHOLE parse tree, each node
    annotated with the Value it computed in that mode. Reach for it to see WHERE a
    surprising answer comes from: which sub-expression rounded, overflowed, or lost
    precision, rather than only the rounded result. For just the final value use
    `calculate`; to find the variable value(s) that drive an expression to a root or
    extremum, use `solver`.

    `tree` is a multi-line string, one node per line, indented by depth (root last-
    applied operator at the top, literals at the leaves). Each line is
    `<OPCODE/LITERAL "lexeme"> Value = <value> (<type>[<scale>], <exact|inexact>)`
    followed by ` · `-separated per-mode details: the value in hex (fixed-point as
    M@D with whole-byte digits, `@<scale>` dropped at scale 0; float as raw IEEE-754
    bits), or a rational's decimal approximation. The `<scale>` is the fixed-point
    decimal scale (omitted for modes without one). A fixed-point node that ROUNDED
    its result also carries a final `· rounding <residual> ≈ <approx>` fragment: the
    exact signed residual `stored − true` (a fraction, bounded by half a unit in the
    last place) and its decimal approximation. It is named `rounding`, NOT `error`,
    on purpose — the reply's top-level `error` field is the failure channel, so a
    `rounding` label keeps "this node rounded" from being misread as "this node
    failed". For example `(1 + 1/2) * 3` in fixed-point:

        BINARY_MUL Value = 3 (fixed-point[0], inexact) · hex 0x03
          BINARY_ADD Value = 1 (fixed-point[0], inexact) · hex 0x01
            LITERAL "1" Value = 1 (fixed-point[0], exact) · hex 0x01
            BINARY_DIV Value = 0 (fixed-point[0], inexact) · hex 0x00 · rounding -1/2 ≈ -0.5
              LITERAL "1" Value = 1 (fixed-point[0], exact) · hex 0x01
              LITERAL "2" Value = 2 (fixed-point[0], exact) · hex 0x02
          LITERAL "3" Value = 3 (fixed-point[0], exact) · hex 0x03

    — the `1/2 = 0` leaf (inexact, scale 0) makes plain that fixed-point rounded the
    half away (its `rounding -1/2` is the exact half discarded), so the product is 3,
    not 4.5, and every node above it inherits the inexactness. Those ancestors show no
    `rounding` fragment: they introduced no rounding of their own, only carried the
    leaf's. (Raise min_fixed_point_precision, or use a different mode, to keep those
    digits.)

    On success `tree` is the rendering and `error` is null; on a bad mode, an invalid
    min_fixed_point_precision, or a malformed/erroring expression, `tree` is null and
    `error` carries the message (the same messages `calculate` returns).
    """
    node, _selected, _value, error = _evaluate_request(expression, mode, min_fixed_point_precision)
    if error is not None:
        return {"tree": None, "error": error}
    assert node is not None  # error is None means the tree evaluated
    return {"tree": node.pretty(), "error": None}


# 43.7: the floor a fixed-point search falls back to when the caller omits
# min_fixed_point_precision. A search at scale 0 would floor the variable to whole
# numbers and miss every non-integer solution (39); defaulting to sub-unit resolution
# lets the bare fixed-point call just work. 9 matches the value the old refusal hinted.
_DEFAULT_SOLVER_FLOOR = 9


@mcp.tool()
def solver(
    expression: Annotated[
        str,
        Field(
            description=(
                "The expression (or newline-separated program whose `name = expr` "
                "lines set constants) to drive to a root or extremum; same grammar "
                "as `calculate`."
            )
        ),
    ],
    variable: Annotated[
        str | None,
        Field(
            description=(
                "SINGLE-unknown form: name of the one variable to search for, used "
                "together with `lower`+`upper`. OPTIONAL: omit it and the solver "
                "auto-detects the sole free name in the expression (it errors only "
                "if there is zero or more than one). Give EXACTLY ONE input form: "
                "this trio, OR `variables` (never both, never neither)."
            )
        ),
    ] = None,
    lower: Annotated[
        float | None,
        Field(
            description=(
                "SINGLE form: lower bound of the search bracket for `variable` "
                "(must be below `upper`). Part of the variable+lower+upper trio; "
                "leave unset when using the `variables` form."
            )
        ),
    ] = None,
    upper: Annotated[
        float | None,
        Field(
            description=(
                "SINGLE form: upper bound of the search bracket for `variable` "
                "(must be above `lower`). Part of the variable+lower+upper trio; "
                "leave unset when using the `variables` form."
            )
        ),
    ] = None,
    variables: Annotated[
        dict[str, list[float]] | None,
        Field(
            description=(
                "MULTIPLE-unknown form: dict mapping each unknown name to its "
                '[lower, upper] bracket, e.g. {"x": [0, 5], "y": [-4, 2]}; '
                "requires algorithm='nelder-mead'. Give EXACTLY ONE input form: "
                "this, OR `variable`+`lower`+`upper` (never both, never neither)."
            )
        ),
    ] = None,
    objective: Annotated[
        str | None,
        Field(
            description=(
                "What to search for: 'find-root' (default), 'find-minimum', or 'find-maximum'."
            )
        ),
    ] = None,
    algorithm: Annotated[
        str | None,
        Field(
            description=(
                "Search engine: 'golden-section-search' (default, single-variable), "
                "'brent-parabolic' (single-variable), 'bisection', 'ridders', "
                "'brent-dekker' or 'chandrupatla' (single-variable, find-root only — "
                "bracket a sign change; the last three converge faster than bisection, "
                "and chandrupatla stays fastest on a repeated root), or 'nelder-mead' "
                "(required for the `variables` form)."
            )
        ),
    ] = None,
    mode: Annotated[
        str,
        Field(
            description=(
                "Numeric type the search runs in: 'fixed-point' (default), "
                "'floating-point', or 'rational' — as in `calculate`."
            )
        ),
    ] = "fixed-point",
    min_fixed_point_precision: Annotated[
        int | None,
        Field(
            description=(
                "Floor on fixed-point fractional digits (non-negative integer). In "
                "fixed-point mode, omit it for a default floor of 9 so the search "
                "resolves non-integer solutions instead of flooring the variable to "
                "whole numbers; pass an explicit value for more/fewer decimals, or 0 to "
                "search integers only. Leave null/omit in the other modes, which resolve "
                "sub-unit values natively."
            )
        ),
    ] = None,
) -> dict:
    """Find the value(s) of one or more variables that drive an expression to a root or extremum.

    `solver` takes the SAME expression language as `calculate` — every operator,
    function, literal form, and (crucially) multi-line programs with `name = expr`
    assignments — but instead of evaluating the expression it SEARCHES for the value
    of the unknown(s) that drive the expression to the chosen `objective`:
      - "find-root" (default): find where the expression equals zero. Write an
        equation `f = g` as the expression `f - g` and find its root.
      - "find-minimum" / "find-maximum": find where the expression reaches its
        smallest / largest value within the bracket(s).

    There are two input forms for the unknowns:
      - SINGLE: `variable` + `lower` + `upper` — one unknown searched over the bracket
        `[lower, upper]` (`lower` must be below `upper`). This is the default
        golden-section engine. `variable` may be OMITTED when the expression has
        exactly one free name; the solver then auto-detects it (e.g. `n` in
        `12*n - (450 + 3*n)`), needing only `lower` + `upper`.
      - MULTIPLE: `variables` — a dict mapping each unknown name to its `[lower, upper]`
        bracket, e.g. `{"x": [0, 5], "y": [-4, 2]}`. This needs the Nelder-Mead engine
        (`algorithm="nelder-mead"`), which searches all the unknowns jointly.
    Give exactly one of the two forms. Each unknown must OCCUR in the expression and
    must NOT be assigned by it; every OTHER name is a constant the program sets via an
    assignment line (e.g. `"r = 0.05\\np = 1000\\np * (1 + r)**n - 2000"` solving for
    `n` with `r`, `p` fixed). A name that is neither an unknown nor assigned is an error.

    `objective` (optional) names what to search for — "find-root", "find-minimum", or
    "find-maximum"; omitted, it defaults to "find-root". (The older spellings `solve`,
    `minimise`/`maximise` and their `min`/`max` and American forms are accepted too.)

    `algorithm` (optional) names the search engine — "golden-section-search" (the
    default, single-variable), "brent-parabolic" (single-variable too, parabolic
    interpolation with a golden-section fallback — usually faster on smooth extrema),
    "bisection", "ridders", "brent-dekker" or "chandrupatla" (single-variable, find-root
    ONLY — all four bracket a sign change and need not have straddling endpoints, since
    they scan the bracket for one; bisection halves the bracket, ridders takes a faster
    exponential-fit step, brent-dekker interpolates inverse-quadratically with a
    bisection fallback — the usual library default for a bracketed root — and
    chandrupatla admits that same interpolation under a sharper test, which keeps it at
    bisection's speed on a repeated root where brent-dekker slows to about a third of
    it), or "nelder-mead" (multivariate, a bounds-clamped downhill simplex). The six
    single-variable engines solve only the SINGLE form; the `variables` form requires
    "nelder-mead". (`golden`, `brent`, `bisect`, `ridder`, `brent-root`, `simplex` and a
    few other spellings are accepted too — note bare `brent` names the PARABOLIC
    MINIMISER, while Brent's root method is `brent-dekker`.)

    `mode` and `min_fixed_point_precision` behave as in `calculate` — the search runs
    in that numeric type and the found value is reported in it — with ONE solver-only
    rule: in fixed-point mode (the default) a search at scale 0 would floor the variable
    to whole numbers and miss any non-integer solution, so when min_fixed_point_precision
    is omitted it DEFAULTS to 9 (sub-unit resolution) rather than 0. Pass an explicit
    value for more/fewer decimals, or 0 to search integers only; the other modes
    (floating-point / rational) resolve sub-unit values natively and take no floor. See
    `calculate` and `help` for the shared grammar and modes; if a
    found value or objective looks off, `analyze` shows the per-node parse tree of the
    expression (with the unknowns substituted) to reveal where it rounded or overflowed.

    The search is bounded by a hard 2-second time limit. If it has not converged by
    then it stops and reports the best value reached so far (a find-root that has not
    got close enough to zero in that time is reported as no-solution, naming the limit).

    Returns a dict: `solutions` is a list of `{variable, solution, solution_hex_dump}`,
    one per unknown in input order — each `solution` rendered and marked "(approximate)"
    (the search locates it to a tolerance, never exactly), with its bit-backed hex. For
    the SINGLE form the scalar `variable` / `solution` / `solution_hex_dump` are also
    set (the one unknown); for the MULTIPLE form those scalars are null and `solutions`
    carries every value. `value` is the EXPRESSION evaluated at that solution, annotated
    with its own precision verdict (near zero for find-root, the extremum otherwise),
    and `value_hex_dump` its hex. `mode` is the resolved numeric type; `exact` and
    `precision` describe `value` exactly as in `calculate`. `objective` echoes the
    resolved objective, `algorithm` names the engine used, and `iterations` is how many
    search steps it took. On any failure — a bad mode/precision, a malformed expression,
    an invalid request (no/both input forms, empty bracket, unknown not in the
    expression, golden-section asked for multiple unknowns, unknown objective or
    algorithm), or no solution in the bracket — every data field is null and `error`
    carries the message (a no-solution error reports the closest |expr| it reached); on
    success `error` is null.
    """
    selected, mode_error = _resolve_mode_and_precision(mode, min_fixed_point_precision)
    if mode_error is not None:
        return _solver_error(mode_error)
    assert selected is not None  # mode_error is None means a mode resolved
    if selected is Mode.COMPLEX:
        # The search engines bracket a sign change / order a real objective; complex
        # results have neither, so the solver does not run in complex mode.
        return _solver_error("the solver is real-valued; complex mode is not supported")
    try:
        resolved_objective = resolve_objective(objective)
        resolved_algorithm = resolve_algorithm(algorithm)
    except SolverError as exc:
        return _solver_error(exc.message)
    try:
        node = parser.parse(expression)
    except (LexError, ParseError) as exc:
        return _solver_error(exc.message)
    try:
        # Parse first: auto-detecting an omitted `variable` (43.3) needs the AST.
        unknowns = _resolve_unknowns(variable, lower, upper, variables, resolved_algorithm, node)
    except SolverError as exc:
        return _solver_error(exc.message)
    # 43.7: a fixed-point search at scale 0 floors the variable to whole numbers and
    # misses every non-integer solution (39). Rather than refuse the bare call, default
    # an omitted floor to _DEFAULT_SOLVER_FLOOR so it resolves sub-unit values out of the
    # box; an explicit value (including 0 for integer-only) overrides it. None reaches
    # here only in fixed-point mode — _resolve_mode_and_precision already rejected a
    # non-fixed-point mode that passed a floor — so the other modes take floor 0 unused.
    if selected is Mode.FIXED_POINT and min_fixed_point_precision is None:
        floor = _DEFAULT_SOLVER_FLOOR
    else:
        floor = min_fixed_point_precision or 0
    try:
        for name, lo, hi in unknowns:
            validate_bracket(lo, hi)
            validate_unknown(node, name)
        if resolved_algorithm is Algorithm.NELDER_MEAD:
            result = nelder_mead(node, unknowns, selected, floor, resolved_objective)
        elif resolved_algorithm is Algorithm.BRENT_PARABOLIC:
            # single-variable, like golden-section — _resolve_unknowns guaranteed one
            name, lo, hi = unknowns[0]
            result = brent_parabolic(node, name, lo, hi, selected, floor, resolved_objective)
        elif resolved_algorithm is Algorithm.BISECTION:
            # single-variable root finder — _resolve_unknowns guaranteed one unknown
            name, lo, hi = unknowns[0]
            result = bisection(node, name, lo, hi, selected, floor, resolved_objective)
        elif resolved_algorithm is Algorithm.RIDDERS:
            # single-variable root finder — _resolve_unknowns guaranteed one unknown
            name, lo, hi = unknowns[0]
            result = ridders(node, name, lo, hi, selected, floor, resolved_objective)
        elif resolved_algorithm is Algorithm.BRENT_DEKKER:
            # single-variable root finder — _resolve_unknowns guaranteed one unknown
            name, lo, hi = unknowns[0]
            result = brent_dekker(node, name, lo, hi, selected, floor, resolved_objective)
        elif resolved_algorithm is Algorithm.CHANDRUPATLA:
            # single-variable root finder — _resolve_unknowns guaranteed one unknown
            name, lo, hi = unknowns[0]
            result = chandrupatla(node, name, lo, hi, selected, floor, resolved_objective)
        else:  # golden-section — _resolve_unknowns guaranteed exactly one unknown
            name, lo, hi = unknowns[0]
            result = search(node, name, lo, hi, selected, floor, resolved_objective)
    except SolverError as exc:
        return _solver_error(exc.message)
    except EvalError as exc:  # a constant the program never set (structural, 31.7)
        return _solver_error(exc.message)
    # Report the floor actually in effect (43.7's default included), so a fixed-point
    # result's precision verdict reflects the engaged floor rather than the raw argument.
    reported_floor = floor if selected is Mode.FIXED_POINT else min_fixed_point_precision
    return _solver_reply(result, selected, reported_floor)


def _resolve_unknowns(
    variable: str | None,
    lower: float | None,
    upper: float | None,
    variables: dict[str, list[float]] | None,
    algorithm: Algorithm,
    node: Node,
) -> list[tuple[str, float, float]]:
    """Normalise the two input forms into the ordered `(name, lower, upper)` list (33.14).

    Exactly one form must be given: the scalar `variable` + `lower` + `upper` trio, or
    the `variables` dict of `name -> [lower, upper]`. The `variables` form is
    multivariate and so requires the Nelder-Mead engine — golden-section drives a
    single unknown only. In the single form `variable` may be OMITTED, in which case
    the unknown is auto-detected from `node` as the expression's sole free name (43.3);
    `lower` + `upper` are still required. Raises SolverError on a missing/double form,
    a malformed `variables` entry, golden-section asked for multiple unknowns, or an
    ambiguous/empty auto-detect. (Whether each bracket has width, and whether each name
    occurs in the program, is checked later by validate_bracket / validate_unknown.)
    """
    has_single = variable is not None or lower is not None or upper is not None
    has_multi = variables is not None
    if has_single and has_multi:
        raise SolverError(
            "Give exactly one unknown form: variable + lower + upper (single), or "
            "variables (multiple); not both."
        )
    if has_multi:
        if algorithm is not Algorithm.NELDER_MEAD:
            raise SolverError(
                f"The {algorithm.value!r} algorithm solves a single variable; pass "
                f"algorithm='nelder-mead' to solve for multiple unknowns."
            )
        if not variables:
            raise SolverError("No unknowns given: 'variables' is empty.")
        unknowns: list[tuple[str, float, float]] = []
        for name, bracket in variables.items():
            if len(bracket) != 2:
                raise SolverError(
                    f"Bracket for {name!r} must be a [lower, upper] pair, got {bracket!r}."
                )
            unknowns.append((name, float(bracket[0]), float(bracket[1])))
        return unknowns
    if variable is None:
        variable = autodetect_variable(node)  # 43.3: infer the sole free name
    if lower is None or upper is None:
        raise SolverError(f"No search bracket given: pass lower and upper bounds for {variable!r}.")
    return [(variable, float(lower), float(upper))]


def _solver_reply(result: SolverResult, mode: Mode, min_fixed_point_precision: int | None) -> dict:
    """Render a successful search into the solver tool's reply dict (31.8 / 33.14).

    Factored from `solver` so the same success shape has ONE builder: the tool
    returns it over the wire, and the verbose test trace renders it for an
    engine-level `search()` / `nelder_mead()` call. `solutions` lists every unknown's
    found value (one entry for golden-section, n for Nelder-Mead), each marked
    approximate with its bit-backed hex. The scalar `variable` / `solution` /
    `solution_hex_dump` echo that list ONLY when there is a single unknown (so the 1-D
    reply is unchanged); for multiple unknowns they are null and `solutions` carries
    them all. `value` (the expression at the solution) is annotated with its precision
    verdict exactly as `calculate`; `exact` / `precision` describe that value.
    """
    value = result.value
    solutions = [
        {
            "variable": name,
            "solution": f"{found.to_string()} (approximate)",
            "solution_hex_dump": found.hex_dump(),
        }
        for name, found in result.solutions
    ]
    single = solutions[0] if len(solutions) == 1 else None
    return {
        "variable": single["variable"] if single else None,
        "solution": single["solution"] if single else None,
        "solution_hex_dump": single["solution_hex_dump"] if single else None,
        "solutions": solutions,
        "value": _annotate(
            value.to_string(), value.exact, value.precision(), None, min_fixed_point_precision
        ),
        "value_hex_dump": value.hex_dump(),
        "mode": mode.value,
        "exact": value.exact,
        "precision": value.precision(),
        "objective": result.objective.value,
        "algorithm": result.algorithm,
        "iterations": result.iterations,
        "error": None,
    }


def _solver_error(message: str) -> dict:
    """A solver reply carrying only an error — every data field null (31.8).

    Mirrors `_error` for calculate: the same key set as the success reply so the
    shape never varies, with solution / solutions / value / mode / objective /
    algorithm / iterations all null and the message in `error`.
    """
    return {
        "variable": None,
        "solution": None,
        "solution_hex_dump": None,
        "solutions": None,
        "value": None,
        "value_hex_dump": None,
        "mode": None,
        "exact": None,
        "precision": None,
        "objective": None,
        "algorithm": None,
        "iterations": None,
        "error": message,
    }


def _error(message: str) -> dict:
    """A calculate result carrying only an error — every value field null (27.1/27.5).

    value/value_hex_dump/mode/exact/precision/offered_precision/values are all null:
    no mode ever resolved (an unknown mode never produced one), no value to dump, and
    nothing to offer or list. Same key set as the success reply so the shape never varies.
    """
    return {
        "value": None,
        "value_hex_dump": None,
        "mode": None,
        "exact": None,
        "precision": None,
        "offered_precision": None,
        "values": None,
        "error": message,
    }


# 44.1: the floor a fixed-point fit falls back to when the caller omits
# min_fixed_point_precision. As in the solver (_DEFAULT_SOLVER_FLOOR), a fit at scale 0
# would floor every fitted parameter to a whole number and ruin a non-integer slope or
# intercept; defaulting to sub-unit resolution lets the bare fixed-point call just work.
_DEFAULT_FIT_FLOOR = 9


@mcp.tool()
def curve_fit(
    x: Annotated[
        list[float],
        Field(
            description=(
                "The x coordinates of the observations — a list of numbers, the same "
                "length as `y` and at least two points long."
            )
        ),
    ],
    y: Annotated[
        list[float],
        Field(
            description=(
                "The y coordinates of the observations — a list of numbers paired with "
                "`x` (same length, at least two points)."
            )
        ),
    ],
    mode: Annotated[
        str,
        Field(
            description=(
                "Numeric type the fit runs in: 'fixed-point' (default), "
                "'floating-point', or 'rational' — as in `calculate`."
            )
        ),
    ] = "fixed-point",
    min_fixed_point_precision: Annotated[
        int | None,
        Field(
            description=(
                "Floor on fixed-point fractional digits (non-negative integer). In "
                "fixed-point mode, omit it for a default floor of 9 so the fitted "
                "parameters keep sub-unit precision instead of rounding to whole "
                "numbers; pass an explicit value for more/fewer decimals. Leave "
                "null/omit in the other modes, which carry no decimal scale."
            )
        ),
    ] = None,
) -> dict:
    """Fit curve(s) to paired (x, y) data and report each fitted equation with its error.

    Reach for `curve_fit` when you have observations and want an EQUATION that describes
    them; use `calculate` to evaluate an equation you already have, or `solver` to find the
    input that drives a known equation to a target. For example, `x=[1, 1.5, 2]`,
    `y=[2, 5.8, 8.9]` comes back with the line `6.9*x - 4.78…` ranked first by its residual
    error, so you can paste that equation straight into `calculate` to predict new points.

    `curve_fit` takes the observations as two equal-length lists — `x` and `y` — and estimates,
    for every curve form it knows, the parameter values that best match the data in the
    least-squares sense, reporting each form's fitted equation and the residual error
    (the sum of squared residuals). The curve library holds the straight line `a*x + b`,
    the quadratic `a*x**2 + b*x + c`, the cubic `a*x**3 + b*x**2 + c*x + d`, the power law
    `a*x**b`, the exponential `a*exp(b*x)`, the logarithm `a + b*ln(x)`, the square root
    `a*sqrt(x) + b`, the reciprocal `a/x + b`, the sinusoid `a*sin(b*x + c) + d`, the
    gaussian `a*exp(-(x-b)**2/(2*c**2))`, the saturation `x/(a*x + b)` (Michaelis-Menten) and
    the hyperbolic `1/(a*x + b)` (44.2.1-44.2.12, the complete library). Every form but the
    sinusoid is fitted in CLOSED FORM — the polynomials
    and the affine logarithm/square-root/reciprocal forms by the exact normal equations, the
    power and exponential laws by a log-linearisation (`ln y = ln a + b·ln x` for the power,
    `ln y = ln a + b·x` for the exponential), the gaussian by Caruana's method (logs to
    the quadratic `ln y = p2·x**2 + p1·x + p0`, fitted by the normal equations, then the
    peak/centre/width are recovered from its coefficients), and the saturation and hyperbolic
    by a reciprocal-line transform (the double-reciprocal Lineweaver-Burk line
    `1/y = a + b·(1/x)` for the saturation, the line `1/y = a·x + b` for the hyperbolic). The
    sinusoid alone has no
    closed-form solution, so it is fitted by an ITERATIVE frequency search: it is non-linear
    only in the frequency `b`, so for each candidate `b` the best amplitude/phase/offset is a
    linear sub-fit, and a coarse scan over a data-derived frequency range (then a
    golden-section refinement) picks the `b` minimising the residual. A form that cannot fit
    the data is dropped, not fatal: the power and logarithm need `x > 0`, the square root
    `x >= 0`, the reciprocal `x != 0`, the exponential, power and gaussian `y > 0` (and the
    gaussian needs bell-shaped, downward-parabola data), the saturation `x != 0` and `y != 0`,
    the hyperbolic `y != 0`; the power, exponential, logarithm,
    square-root, sinusoid and gaussian forms also drop in rational mode, which cannot represent
    their irrational logs/roots/sines/exp (the saturation and hyperbolic, being pure
    reciprocals, stay exact in rational); and a polynomial needs enough distinct `x` (the
    sinusoid four). The forms that fit are ranked by their residual error — least error first —
    and only the best three are returned (fewer when fewer forms fit).

    `mode` and `min_fixed_point_precision` behave as in `calculate` — the whole fit runs
    in that numeric type, so the parameters and error are exact in `rational`, rounded at
    the scale in `fixed-point`, and native doubles in `floating-point`. As in the
    `solver`, a fixed-point fit at scale 0 would round every fitted parameter to a whole
    number, so when `min_fixed_point_precision` is omitted it DEFAULTS to 9 (sub-unit
    resolution); pass an explicit value for more/fewer decimals. Complex mode is not
    supported (the fit is real-valued).

    Returns a dict: `fits` is the ranked list of fitted forms (best first, at most three),
    each `{form, equation, parameters, fit_error, fit_error_hex_dump, exact, precision}` —
    `form` names the curve, `equation` is the model with its parameters substituted (over
    the variable `x`, so it can be pasted into `calculate`), `parameters` is a list of
    `{name, value, value_hex_dump}` for each fitted coefficient (each `value` annotated
    with its own precision verdict), and `fit_error` is the residual error annotated the
    same way with its hex dump; that fit's own `exact` and `precision` describe its error.
    `mode` is the resolved numeric type, and the top-level `exact` / `precision` describe
    the BEST fit's error (the headline number), as in `solver`. On any failure — a bad
    mode/precision, mismatched or too-short data, or data that cannot support the form
    (every x equal, so a line has no slope) — `fits`/`mode`/`exact`/`precision` are null and
    `error` carries the message; on success `error` is null.
    """
    selected, mode_error = _resolve_mode_and_precision(mode, min_fixed_point_precision)
    if mode_error is not None:
        return _fit_error(mode_error)
    assert selected is not None  # mode_error is None means a mode resolved
    if selected is Mode.COMPLEX:
        # A fit minimises a real residual; complex data has no ordering to minimise.
        return _fit_error("the fit is real-valued; complex mode is not supported")
    if len(x) != len(y):
        return _fit_error(f"x and y must have the same length, got {len(x)} and {len(y)}.")
    if len(x) < 2:
        return _fit_error(f"Need at least two (x, y) points to fit a curve, got {len(x)}.")
    # 44.1: a fixed-point fit at scale 0 floors every parameter to a whole number; default
    # an omitted floor to _DEFAULT_FIT_FLOOR so the bare call resolves sub-unit values. None
    # reaches here only in fixed-point mode (_resolve_mode_and_precision rejected a floor in
    # the other modes), so they take floor 0 unused.
    if selected is Mode.FIXED_POINT and min_fixed_point_precision is None:
        floor = _DEFAULT_FIT_FLOOR
    else:
        floor = min_fixed_point_precision or 0
    try:
        results = fit_all(x, y, selected, floor)
    except FitError as exc:
        return _fit_error(exc.message)
    reported_floor = floor if selected is Mode.FIXED_POINT else min_fixed_point_precision
    best_error = results[0].error  # results are ranked best-first; the headline is the best fit's
    return {
        "fits": [_fit_entry(result, reported_floor) for result in results],
        "mode": selected.value,
        "exact": best_error.exact,
        "precision": best_error.precision(),
        "error": None,
    }


def _fit_entry(result: FitResult, min_fixed_point_precision: int | None) -> dict:
    """Render one FitResult into the fit reply's per-form object (44.1 / 44.5).

    Each parameter and the error are annotated with their own precision verdict exactly as
    `calculate` annotates a value, so a fitted coefficient is never mistaken for exact when
    the mode rounded it. `exact` / `precision` describe the error value (the fit's headline
    number).
    """
    error = result.error
    return {
        "form": result.form,
        "equation": result.equation,
        "parameters": [
            {
                "name": name,
                "value": _annotate(
                    value.to_string(),
                    value.exact,
                    value.precision(),
                    None,
                    min_fixed_point_precision,
                ),
                "value_hex_dump": value.hex_dump(),
            }
            for name, value in result.parameters
        ],
        "fit_error": _annotate(
            error.to_string(), error.exact, error.precision(), None, min_fixed_point_precision
        ),
        "fit_error_hex_dump": error.hex_dump(),
        "exact": error.exact,
        "precision": error.precision(),
    }


def _fit_error(message: str) -> dict:
    """A fit reply carrying only an error — every data field null (44.1).

    Mirrors `_solver_error` / `_error`: the same key set as the success reply so the shape
    never varies, with `fits` / `mode` / `exact` / `precision` null and the message in `error`.
    """
    return {"fits": None, "mode": None, "exact": None, "precision": None, "error": message}


# FUTURE (SA.1): opt-in toolset gating goes here. When the feature toolsets
# (numeric | bits | eval) exist, read the enabled set from config/env at
# import time and register only those tools, mirroring mcp-tmux's opt-in
# toolset pattern. Until then every tool below registers unconditionally.
