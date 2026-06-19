# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""In-server language reference, rendered as terse section text (TODO 21).

The `help` MCP tool reads this to drive the evaluator correctly. Every section
is BUILT FROM the live engine — the type set from the ``Mode`` enum, the operator
set and precedence from the parser's binding table, the literal bases from the
lexer — so the reference cannot drift from what the code actually accepts. That
mirroring is the whole point, so this module reaches into the grammar's private
constants on purpose. Text is facts only; the model fills in the prose.

Unknown sections do not error: ``render`` returns the list of valid sections, so
a wrong guess is self-correcting.
"""

from collections.abc import Callable

from mcp_abacus.expr.lexer import _BASE_PREFIXES
from mcp_abacus.expr.nodes import FUNCTION_ARITIES, FUNCTION_HELP, UNARY_OPS
from mcp_abacus.expr.parser import _BINDING_POWER, _POWER_OPS
from mcp_abacus.expr.value import MODE_ALIASES, MODE_HELP, Mode


def _types_section() -> str:
    # m.value is the wire name ("floating-point"); MODE_HELP is the single source for
    # the one-liner — a missing entry raises, forcing co-update with the enum. The
    # accepted aliases (41.11) are derived from the live MODE_ALIASES map and grouped
    # per mode, so every spelling resolve_mode honours is advertised here and the list
    # cannot drift from what the engine actually accepts — `decimal` in particular
    # resolving to fixed-point is now visible rather than a silent surprise.
    aliases: dict[Mode, list[str]] = {m: [] for m in Mode}
    for name, mode in MODE_ALIASES.items():
        aliases[mode].append(name)
    lines = []
    for m in Mode:
        suffix = f"; aliases: {', '.join(aliases[m])}" if aliases[m] else ""
        lines.append(f"{m.value} — {MODE_HELP[m]}{suffix}")
    return "\n".join(lines)


def _language_section() -> str:
    # Group the table operators by binding power, preserving the table's own
    # loose->tight insertion order; unary and power sit outside the table (plain
    # descent in the parser) so their rung order/associativity is fixed here.
    levels: dict[int, list[str]] = {}
    for op, bp in _BINDING_POWER.items():
        levels.setdefault(bp, []).append(op)

    rungs = [f"  {' '.join(ops):<12} left-assoc" for _bp, ops in sorted(levels.items())]
    rungs.append(f"  {' '.join(sorted(UNARY_OPS)):<12} unary prefix")
    rungs.append(
        f"  {' '.join(sorted(_POWER_OPS)):<12} right-assoc; tighter than unary minus "
        "(-2**2 == -(2**2)); exponent may be unary (2**-3)"
    )

    bases = " ".join(sorted({p.lower() for p in _BASE_PREFIXES}))
    return "\n".join(
        [
            "operators (loose to tight):",
            *rungs,
            "",
            "** is POWER; ^ & | are BITWISE (^ is XOR, not power) and ~ is bitwise",
            "NOT. Bitwise ops work in EVERY type, on that type's own stored bits:",
            "the 64-bit IEEE-754 pattern (float), the scale-aligned mantissa",
            "(fixed-point), numerator/denominator (rational). Both operands of a",
            "binary op must share the same type; there is no implicit promotion.",
            "",
            "literals:",
            "  decimal       12  3.14  .5  1e3  2.5e-4",
            f"  base integer  {bases} prefixes; that integer in every type (0x1F 0b1010 0o17)",
            "  fixed-point   <base-int>@<decimals> == M x 10^-D; M MUST be base-prefixed",
            "                (0x/0o/0b) — a DECIMAL mantissa is INVALID: 123@2 and 123.45@2",
            "                both error; write a decimal value as its digits (123.45), never @.",
            "                e.g. 0x59682F00@9 = 1.5; 0xDE0B6B3A7640000@18 = 1 ETH",
            "  grouping      ( )",
            "",
            "variables & statements:",
            "  name          identifier [A-Za-z_][A-Za-z0-9_]*; a bare name reads a",
            "                variable (error if never assigned). name(...) is a call, not a read.",
            "  assignment    name = expr  binds name to expr's value; loosest precedence,",
            "                statement-level only (no nesting). The assignment's OWN value is",
            "                expr's, so `x = 2 + 3` yields 5 and also binds x.",
            "  statements    one per line, separated by newlines; run in order, sharing one",
            "                variable scope, so a later line sees earlier bindings. A program's",
            "                value is the LAST statement's (earlier lines run for their bindings).",
            "                Scope is per call — bindings do not persist across calls.",
            "  comment       # to end of line is ignored (the newline still separates",
            "                statements); a whole line may be just a comment.",
        ]
    )


# Param names for a generated signature — purely illustrative; the call shape
# (how many args, whether variadic) is the fact, read off FUNCTION_ARITIES.
_PARAM_NAMES = ("x", "y", "z", "w", "v", "u")


def _signature(name: str, lo: int, hi: int | None) -> str:
    # Render a function's call shape from its arity range. A variadic tail (hi is
    # None) shows the required params then "…" (one or more); a fixed arity shows
    # exactly its params; a bounded range marks the optional tail in [brackets].
    if hi is None:
        return f"{name}({', '.join((*_PARAM_NAMES[:lo], '…'))})"
    if lo == hi:
        return f"{name}({', '.join(_PARAM_NAMES[:lo])})"
    required = ", ".join(_PARAM_NAMES[:lo])
    optional = ", ".join(_PARAM_NAMES[lo:hi])
    return f"{name}({required}[, {optional}])"


def _functions_section() -> str:
    # Built from FUNCTION_ARITIES, the same live registry the parser validates
    # against, so the list cannot drift from what the engine actually accepts. No
    # precedence among functions, so list them alphabetically. Each row is the call
    # shape plus its one-line semantics from FUNCTION_HELP (the parallel registry);
    # the descriptions are aligned in a column for legibility.
    names = sorted(FUNCTION_ARITIES)
    signatures = {name: _signature(name, *FUNCTION_ARITIES[name]) for name in names}
    width = max(len(sig) for sig in signatures.values())
    rows = [f"  {signatures[name]:<{width}}  — {FUNCTION_HELP[name]}" for name in names]
    return "\n".join(
        [
            "functions (called as name(arg, ...); each argument evaluates in the active",
            'type, like an operator; "…" means one or more):',
            *rows,
        ]
    )


def _solver_section() -> str:
    # Built from the solver's live objective enum and aliases, so the section cannot
    # drift from what the tool accepts. Imported LOCALLY: the solver is a higher-level
    # sibling that depends on this expr subpackage, so a module-level import here would
    # invert the layering (and risk a load-time cycle) — deferring it to call time
    # keeps expr free of an edge back to solver.
    from mcp_abacus.solver import _OBJECTIVE_ALIASES, Algorithm, Objective

    objectives = " | ".join(o.value for o in Objective)
    aliases = ", ".join(f"{name}={o.value}" for name, o in _OBJECTIVE_ALIASES.items())
    algorithms = " | ".join(a.value for a in Algorithm)
    return "\n".join(
        [
            "solver tool — find the value(s) of one or more variables that drive an",
            "expression to a target over required bracket(s). SAME expression language as",
            "calculate (operators, functions, literals, multi-line `name = expr` programs —",
            "see the language section); it SEARCHES for the unknown(s) rather than evaluating.",
            "",
            "arguments:",
            "  expression  the program to drive; calculate's grammar.",
            "  variable    the single unknown (SINGLE form). Must OCCUR in the expression",
            "              and must NOT be assigned by it; every OTHER name is a constant",
            "              the program sets via an assignment line.",
            "  lower upper the search bracket [lower, upper]; lower must be below upper.",
            "  variables   the MULTIPLE form: a dict name -> [lower, upper], e.g.",
            '              {"x": [0, 5], "y": [-4, 2]}. Needs algorithm nelder-mead.',
            "              Give EXACTLY ONE of (variable+lower+upper) or variables.",
            f"  objective   what to search for: {objectives}.",
            "              Omitted -> find-root.",
            f"              (aliases: {aliases})",
            f"  algorithm   the search engine: {algorithms}.",
            "              Omitted -> golden-section-search (single variable only).",
            "  mode        numeric type the search runs in (see types); default fixed-point.",
            "  min_fixed_point_precision  fixed-point decimal floor, exactly as in calculate.",
            "",
            "objectives:",
            "  find-root     find where the expression equals zero — a root. Write an",
            "                equation f = g as the expression f - g. 'No solution' when zero",
            "                is not reached in the bracket (the closest |expr| is reported).",
            "  find-minimum  find where the expression is smallest within the bracket(s).",
            "  find-maximum  find where the expression is largest within the bracket(s).",
            "",
            "algorithms:",
            "  golden-section-search  single unknown, shrinks the 1-D bracket. The default.",
            "  brent-parabolic        single unknown, parabolic interpolation with a golden-",
            "                         section fallback; usually faster on smooth extrema.",
            "  bisection              single unknown, find-root ONLY: scans the bracket for a",
            "                         sign change and halves it. Robust; the endpoints need",
            "                         not already straddle zero.",
            "  ridders                single unknown, find-root ONLY: same sign-change bracket",
            "                         as bisection but an exponential-fit step; converges faster.",
            "  nelder-mead            n unknowns, walks a downhill simplex; bounds-clamped to",
            "                         each bracket. The only engine for the variables form.",
            "",
            "reply: solutions (a list of {variable, solution, solution_hex_dump}, one per",
            "unknown, each marked approximate); for the SINGLE form the scalar variable /",
            "solution / solution_hex_dump are also set. value (the expression AT the",
            "solution — near zero for find-root, the extremum otherwise) + value_hex_dump;",
            "mode, exact, precision (describing value, as in calculate); objective,",
            "algorithm, iterations; error. On failure every data field is null and error",
            "carries the message.",
        ]
    )


# section name -> (one-line descriptor for the index, builder). Add a section by
# adding a row; the call shape never changes.
_SECTIONS: dict[str, tuple[str, Callable[[], str]]] = {
    "types": ("numeric types this build supports", _types_section),
    "language": ("expression grammar: operators, precedence, literals", _language_section),
    "functions": ("the callable functions and their argument counts", _functions_section),
    "solver": ("the solver tool: find a root or extremum over a bracket", _solver_section),
}


def sections() -> tuple[str, ...]:
    """The valid section names, in registry order."""
    return tuple(_SECTIONS)


def index() -> str:
    """A one-line-per-section listing of the available reference sections."""
    return "\n".join(f"{name} — {desc}" for name, (desc, _) in _SECTIONS.items())


def render(section: str) -> str:
    """Return ``section``'s reference text, or the valid-section list if unknown."""
    entry = _SECTIONS.get(section)
    if entry is None:
        listing = "\n".join(f"  {name} — {desc}" for name, (desc, _) in _SECTIONS.items())
        return f"unknown section {section!r}. valid sections:\n{listing}"
    return entry[1]()
