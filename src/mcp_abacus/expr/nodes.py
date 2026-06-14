# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Frozen-dataclass AST nodes for the expression engine (TODO 17 + 18).

In-node evaluation: each node implements evaluate(mode) and stores its own
result (TODO 18); the lexer/parser arrive with item 20. Number lexemes are
the UNSIGNED raw source text — sign is a UnaryOp; literal well-formedness is
the lexer's contract, not validated here.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from mcp_abacus.expr.value import EvalContext, Mode, Value, _lexeme_scale

UNARY_OPS: frozenset[str] = frozenset({"+", "-", "~"})  # ~ is bitwise NOT (24.3.2)
# ** is POWER; ^ & | are bitwise (24.3.2 — ^ is XOR, not power).
BINARY_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", "//", "%", "**", "&", "|", "^"})

# 26.8: the analyze-tree label spells each operator out as an UPPER_SNAKE opcode
# (BINARY_ADD, UNARY_NEG) instead of the bare symbol, so the printout reads as a
# named operation. One entry per operator in the frozensets above — a missing
# entry is a hard KeyError when the tree renders, never a silent fallback.
_UNARY_OPCODES: dict[str, str] = {"+": "UNARY_POS", "-": "UNARY_NEG", "~": "UNARY_NOT"}
_BINARY_OPCODES: dict[str, str] = {
    "+": "BINARY_ADD",
    "-": "BINARY_SUB",
    "*": "BINARY_MUL",
    "/": "BINARY_DIV",
    "//": "BINARY_FLOORDIV",
    "%": "BINARY_MOD",
    "**": "BINARY_POW",
    "&": "BINARY_AND",
    "|": "BINARY_OR",
    "^": "BINARY_XOR",
}

# Value has no operator dunders by design (19.5) — dispatch to its named methods.
_UNARY_FUNCS: dict[str, Callable[[Value], Value]] = {
    "+": Value.pos,
    "-": Value.neg,
    "~": Value.bitnot,
}
_BINARY_FUNCS: dict[str, Callable[[Value, Value], Value]] = {
    "+": Value.add,
    "-": Value.sub,
    "*": Value.mul,
    "/": Value.div,
    "//": Value.floordiv,
    "%": Value.mod,
    "**": Value.pow,  # our ** is power, routed to Value.pow — never a Python dunder (18.2)
    "&": Value.bitand,
    "|": Value.bitor,
    "^": Value.bitxor,  # ^ is XOR (24.3.2), routed to Value.bitxor — never a dunder (18.2)
}

# The function set (22.3): name -> the per-mode Value method it dispatches to
# (19.5/22.4), parallel to the operator tables above. THIS registry is the single
# source — adding a function is one entry here plus its Value method, the same
# two-step shape as adding an operator. The function set is not the lexer's or
# parser's concern; both consult this table (the parser via FUNCTION_ARITIES).
_FUNCS: dict[str, Callable[..., Value]] = {
    "abs": Value.abs_,  # 22.4.1 — exact magnitude in every mode (shape of neg())
    "sqrt": Value.sqrt,  # 22.4.2 — irrational, inexact except on the mode's grid
    "pow": Value.pow,  # 28.20 — binary; the call form of **, reuses Value.pow (fixed-arity 2)
    "sin": Value.sin,  # 28.10 — transcendental; fixed-point Taylor series, else inexact/refuse
    "cos": Value.cos,  # 28.11 — sin's machinery; even Taylor series, else inexact/refuse
    "tan": Value.tan,  # 28.12 — sin/cos; fixed-point divides the two series, else inexact/refuse
    "cot": Value.cot,  # 28.13 — cos/sin; mirror of tan, undefined where sin = 0, else inexact
    "log": Value.log,  # 28.17 — NATURAL log; base-10 reduce + atanh series, else inexact/refuse
    "ln": Value.log,  # 28.17 — alias of log (the canonical natural-log spelling)
    "log10": Value.log10,  # 28.18 — base-10 log; ln(x)/ln(10), exact on powers of ten
    "sum": Value.sum_,  # 28.5 — variadic; repeated + via reduce, exact in every mode
    "product": Value.product,  # 28.6 — variadic; repeated * via reduce, may round (covering scale)
    "avg": Value.avg,  # 28.4 — variadic; sum / count, follows the mode's / rule
    "max": Value.max_,  # 28.2 — variadic; selection (largest), exact, carries the operand verbatim
    "min": Value.min_,  # 28.3 — variadic; mirror of max (smallest)
    "median": Value.median,  # 28.7 — variadic; order-only, odd selects (exact), even averages
    "variance": Value.variance,  # 28.8 — variadic; population sum-of-squared-deviations / n
    "stddev": Value.stddev,  # 28.9 — variadic; sqrt(variance), inherits sqrt's per-mode story
}

# The nullary set (29.2): zero-argument calls like pi(). A SECOND registry, parallel
# to _FUNCS but a different callable KIND — these are NOT operand-methods. With no
# operand to carry the mode, a nullary takes the per-run EvalContext (29.1) and
# returns a Value; FuncCall dispatches to it with the context instead of evaluated
# operands. Same single-source rule as _FUNCS: a nullary is one entry here plus its
# Value constructor. Arity is a fixed 0 (set in FUNCTION_ARITIES), NOT read off the
# signature — the ctx parameter is engine-injected, never a user argument.
_NULLARY_FUNCS: dict[str, Callable[[EvalContext], Value]] = {
    "pi": Value.pi,  # 29.2 — circle constant; per-mode value 29.3
    "e": Value.e,  # 29.2 — Euler's number; per-mode value 29.3
}


def _arity_of(func: Callable[..., Value]) -> tuple[int, int | None]:
    """Allowed argument count (min, max) READ OFF a method's signature (22.2).

    self IS the first operand, so a unary method like sqrt(self) is (1, 1) and a
    binary one (self, other) is (2, 2). A ``*args`` (VAR_POSITIONAL) tail makes the
    max unbounded — sum_(self, *others) is (1, None) — so variadic funcs declare a
    MINIMUM, not a fixed count. Reading it off the method keeps it from drifting
    from _FUNCS, the same invariant the single-count form had.
    """
    params = inspect.signature(func).parameters.values()
    required = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    variadic = any(p.kind is p.VAR_POSITIONAL for p in params)
    return (required, None if variadic else required)


# Arity range per function (min, max|None). The parser (22.2) and FuncCall both
# validate a call's argument count against this without importing the methods.
# Nullaries (29.2) join with a fixed (0, 0) — their ctx parameter is engine-
# injected, so the count is NOT read off the signature like the operand-methods.
FUNCTION_ARITIES: dict[str, tuple[int, int | None]] = {
    **{name: _arity_of(func) for name, func in _FUNCS.items()},
    **{name: (0, 0) for name in _NULLARY_FUNCS},
}


def _describe_arity(lo: int, hi: int | None) -> str:
    """Human phrasing of an arity range, for the wrong-count error messages."""
    if hi is None:
        return f"at least {lo} argument(s)"
    if hi == lo:
        return f"{lo} argument(s)"
    return f"{lo} to {hi} argument(s)"


def _arity_ok(name: str, count: int) -> bool:
    """Whether ``count`` arguments is allowed for the registered function ``name``."""
    lo, hi = FUNCTION_ARITIES[name]
    return lo <= count and (hi is None or count <= hi)


class EvalError(Exception):
    """Evaluation failure, carrying the 1-based source line of the failing node."""

    def __init__(self, message: str, line: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


def _validate_line(line: int) -> None:
    if line < 1:
        raise ValueError(f"line must be >= 1, got {line}")


class Node(ABC):
    """Abstract base of all AST nodes."""

    __slots__ = ()

    # Declared here for the shared evaluate()/pretty() machinery; every
    # concrete subclass defines both as dataclass fields.
    line: int
    value: Value | None

    @abstractmethod
    def _label(self) -> str:
        """One-line node header: kind + op/quoted-lexeme (no line number — 26.4)."""

    @abstractmethod
    def _children(self) -> tuple["Node", ...]:
        """Child nodes in source order."""

    @abstractmethod
    def _evaluate(self, ctx: EvalContext) -> Value:
        """Compute this node's Value under the run context (no storing, no wrapping).

        Children recurse through ``_walk`` so the SAME context flows down the whole
        tree; ``ctx`` carries the mode and the fixed-point precision floor (29.1).
        """

    def evaluate(self, mode: Mode, min_fixed_point_precision: int = 0) -> Value:
        """Evaluate the subtree in ONE mode; store and return this node's Value.

        Re-evaluating under another mode OVERWRITES the stored values (18.5).
        Arithmetic failures raise EvalError carrying the failing node's line.

        ``min_fixed_point_precision`` (25.2.1) is the fixed-point scale floor;
        like ``mode`` it is threaded down and only consumed on the literals, where
        it raises each fixed-point operand to at least that many decimals so the
        scale propagates through the calculation. Defaults to 0 (no floor); it is
        a no-op outside fixed-point mode, which has no decimal scale.

        This is the public entry: it bundles the run state into the per-run
        EvalContext (29.1) and walks the tree threading that one object down,
        rather than passing the state to every node or reaching for a module global.
        In FIXED_POINT it first PRE-WALKS the tree once to derive the nullary scale
        (29.3) — the floor raised to the widest literal scale — so a nullary like
        ``pi()``, which has no operand to carry a scale, is computed to match the
        precision of the literals it shares the expression with.
        """
        nullary_precision = min_fixed_point_precision
        if mode is Mode.FIXED_POINT:
            nullary_precision = max(min_fixed_point_precision, self._max_literal_scale())
        ctx = EvalContext(
            mode=mode,
            min_fixed_point_precision=min_fixed_point_precision,
            nullary_precision=nullary_precision,
        )
        return self._walk(ctx)

    def _max_literal_scale(self) -> int:
        """Largest written decimal scale among the literals in this subtree (29.3).

        The pre-walk behind ``nullary_precision``: every node folds the max over its
        children, ``Number`` overriding with its own lexeme scale. 0 for a subtree
        with no literal (e.g. a bare ``pi()``), leaving the nullary scale at the floor.
        """
        return max((child._max_literal_scale() for child in self._children()), default=0)

    def _walk(self, ctx: EvalContext) -> Value:
        """Evaluate this node under a shared context; store and return its Value.

        The recursion step every node funnels through (children call it on their
        own children), so one EvalContext built at the top reaches the whole walk.
        """
        try:
            result = self._evaluate(ctx)
        except ArithmeticError as exc:
            # A child's EvalError is not ArithmeticError, so it propagates
            # untouched — the line stays the innermost failing node's (18.4).
            raise EvalError(str(exc), line=self.line) from exc
        object.__setattr__(self, "value", result)  # designated slot on a frozen node (18.5)
        return result

    def pretty(self) -> str:
        """Multi-line indented tree view, one node per line."""
        return "\n".join(self._pretty_lines(0))

    def _pretty_lines(self, depth: int) -> list[str]:
        label = "  " * depth + self._label()
        if self.value is not None:
            # 18.6/26: show what each PART got — value, type, scale, exactness, and
            # the mode's hex / exact-decimal / approximation details (Value.describe).
            label += f" Value = {self.value.describe()}"
        lines = [label]
        for child in self._children():
            lines.extend(child._pretty_lines(depth + 1))
        return lines


@dataclass(frozen=True, slots=True)
class Number(Node):
    """A numeric literal, kept exactly as written in the source (unsigned)."""

    lexeme: str
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.lexeme:
            raise ValueError("Number lexeme must not be empty")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.2/26.8: "LITERAL" with the source lexeme quoted, so it reads as the text.
        return f'LITERAL "{self.lexeme}"'

    def _children(self) -> tuple[Node, ...]:
        return ()

    def _max_literal_scale(self) -> int:
        # The one node that carries a scale: read it off the verbatim lexeme (29.3).
        return _lexeme_scale(self.lexeme)

    def _evaluate(self, ctx: EvalContext) -> Value:
        # The mode only ever lands here, on the literals (18.3) — and so does the
        # min_fixed_point_precision floor (25.2.1), for the same reason; both ride
        # the context down and are unpacked at this leaf.
        return Value.from_lexeme(self.lexeme, ctx.mode, ctx.min_fixed_point_precision)


@dataclass(frozen=True, slots=True)
class UnaryOp(Node):
    """A unary sign operator applied to an operand."""

    op: str
    operand: Node
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.op not in UNARY_OPS:
            raise ValueError(f"unknown unary operator: {self.op!r}")
        _validate_line(self.line)

    def _label(self) -> str:
        return _UNARY_OPCODES[self.op]

    def _children(self) -> tuple[Node, ...]:
        return (self.operand,)

    def _evaluate(self, ctx: EvalContext) -> Value:
        return _UNARY_FUNCS[self.op](self.operand._walk(ctx))


@dataclass(frozen=True, slots=True)
class BinOp(Node):
    """A binary operator; ``**`` is power and ``^`` is bitwise XOR (24.3.2)."""

    op: str
    left: Node
    right: Node
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.op not in BINARY_OPS:
            raise ValueError(f"unknown binary operator: {self.op!r}")
        _validate_line(self.line)

    def _label(self) -> str:
        return _BINARY_OPCODES[self.op]

    def _children(self) -> tuple[Node, ...]:
        return (self.left, self.right)

    def _evaluate(self, ctx: EvalContext) -> Value:
        # Plain arithmetic on Values — all type-faithful semantics live in Value (18.2).
        return _BINARY_FUNCS[self.op](
            self.left._walk(ctx),
            self.right._walk(ctx),
        )


@dataclass(frozen=True, slots=True)
class FuncCall(Node):
    """A function call like ``sqrt(x)`` or the nullary ``pi()`` (22.3 / 29.2): a
    NAME and its argument subtrees (empty for a nullary).

    Name and arity are validated in the parser (22.2 -> ParseError); the same
    checks here guard direct construction, mirroring UnaryOp/BinOp's op check.
    Evaluation forwards to the dispatched Value callable — an operand-method fed
    the evaluated arguments, or (for a nullary, 29.2) the run context — every
    per-mode semantic living there (18.2), so this node is mode-agnostic.
    """

    name: str
    args: tuple[Node, ...]
    line: int = field(kw_only=True, compare=False)
    value: Value | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.name not in _FUNCS and self.name not in _NULLARY_FUNCS:
            raise ValueError(f"unknown function: {self.name!r}")
        if not _arity_ok(self.name, len(self.args)):
            lo, hi = FUNCTION_ARITIES[self.name]
            raise ValueError(f"{self.name!r} takes {_describe_arity(lo, hi)}, got {len(self.args)}")
        _validate_line(self.line)

    def _label(self) -> str:
        # 26.8: the call reads as CALL with the function name quoted, paralleling LITERAL.
        return f'CALL "{self.name}"'

    def _children(self) -> tuple[Node, ...]:
        return self.args

    def _evaluate(self, ctx: EvalContext) -> Value:
        # A nullary takes the context, not operands (29.2); every other function is
        # an operand-method fed its evaluated arguments. Arity 0 vs the operand path
        # is settled by which registry holds the name.
        if self.name in _NULLARY_FUNCS:
            return _NULLARY_FUNCS[self.name](ctx)
        return _FUNCS[self.name](*(arg._walk(ctx) for arg in self.args))
