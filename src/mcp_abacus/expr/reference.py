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
from mcp_abacus.expr.nodes import FUNCTION_ARITIES, UNARY_OPS
from mcp_abacus.expr.parser import _BINDING_POWER, _POWER_OPS
from mcp_abacus.expr.value import MODE_HELP, Mode


def _types_section() -> str:
    # m.value is the wire name ("floating-point"); MODE_HELP is the single source for
    # the one-liner — a missing entry raises, forcing co-update with the enum.
    return "\n".join(f"{m.value} — {MODE_HELP[m]}" for m in Mode)


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
    # precedence among functions, so list them alphabetically. Semantics are the
    # model's to fill in — only the names and call shapes are stated here.
    rows = [f"  {_signature(name, *FUNCTION_ARITIES[name])}" for name in sorted(FUNCTION_ARITIES)]
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
    from mcp_abacus.solver import _ALGORITHM, _OBJECTIVE_ALIASES, Objective

    objectives = " | ".join(o.value for o in Objective)
    aliases = ", ".join(f"{name}={o.value}" for name, o in _OBJECTIVE_ALIASES.items())
    return "\n".join(
        [
            "solver tool — find the value of ONE variable that drives an expression to",
            "a target over a required bracket. SAME expression language as calculate",
            "(operators, functions, literals, multi-line `name = expr` programs — see the",
            "language section); it SEARCHES for an unknown rather than evaluating.",
            "",
            "arguments:",
            "  expression  the program to drive; calculate's grammar.",
            "  variable    the single unknown. Must OCCUR in the expression and must NOT",
            "              be assigned by it; every OTHER name is a constant the program",
            "              sets via an assignment line.",
            "  lower upper the search bracket [lower, upper]; lower must be below upper.",
            f"  objective   what to search for: {objectives}.",
            "              Omitted -> find-root.",
            f"              (aliases: {aliases})",
            "  mode        numeric type the search runs in (see types); default fixed-point.",
            "  min_fixed_point_precision  fixed-point decimal floor, exactly as in calculate.",
            "",
            "objectives:",
            "  find-root     find where the expression equals zero — a root. Write an",
            "                equation f = g as the expression f - g. 'No solution' when zero",
            "                is not reached in the bracket (the closest |expr| is reported).",
            "  find-minimum  find where the expression is smallest within the bracket.",
            "  find-maximum  find where the expression is largest within the bracket.",
            "",
            "reply: solution (the found value, marked approximate) + solution_hex_dump;",
            "value (the expression AT the solution — near zero for find-root, the extremum",
            "otherwise) + value_hex_dump; mode, exact, precision (describing value, as in",
            f"calculate); objective, algorithm ({_ALGORITHM}), iterations; error. On failure",
            "every data field is null and error carries the message.",
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


def render(section: str) -> str:
    """Return ``section``'s reference text, or the valid-section list if unknown."""
    entry = _SECTIONS.get(section)
    if entry is None:
        index = "\n".join(f"  {name} — {desc}" for name, (desc, _) in _SECTIONS.items())
        return f"unknown section {section!r}. valid sections:\n{index}"
    return entry[1]()
