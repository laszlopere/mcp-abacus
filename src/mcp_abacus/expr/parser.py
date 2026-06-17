# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Recursive-descent parser: source text -> AST nodes (TODO 20.2).

Precedence, loose -> tight (20.2.1 — math convention, NOT Excel; the bitwise
rungs and ~ follow C/Python, 24.3.2):
bitwise OR ``|`` < bitwise XOR ``^`` < bitwise AND ``&`` < additive ``+ -``
< multiplicative ``* / // %`` (all left-assoc) < unary ``+ - ~`` < power ``**``
(RIGHT-assoc and tighter than unary minus: ``-2**2 == -(2**2)``; the exponent may
itself be unary: ``2**-3``). Atoms are numbers, parenthesized expressions,
function calls ``NAME '(' (expr (',' expr)*)? ')'`` (22.2 — tightest binding, so
``sqrt(4)**2`` is ``(sqrt(4))**2``; the argument list may be EMPTY for a nullary
like ``pi()``, 29.2; the call's name and arity are validated here against
nodes.FUNCTION_ARITIES, unknown-name/wrong-arity being parse errors), and bare
NAME variable references (30.4 — a NAME NOT followed by ``(``, distinct from a
call). A bare NAME in nodes.CONSTANT_NAMES (``pi``/``e``, 29.6) is the EXCEPTION:
it parses to the nullary call rather than a Var, so the constant reads like a
literal; those names are reserved, so assigning to one is a parse error. Parens
only shape the tree — no Group node. Above the whole chain sits the
loosest-precedence assignment ``NAME '=' expr`` (30.3), recognised only at
statement level. The left-assoc binary levels run on a Pratt binding-power table;
assignment/unary/power/atom are plain descent.

The top level is a list of newline-separated statements (30.5): a NEWLINE ends a
statement, so an expression spans lines ONLY inside parentheses (where NEWLINE is
treated as insignificant via the paren-depth gate). One statement parses to its
bare node; several wrap in a Sequence.

``^`` is bitwise XOR, NOT power (power is ``**``, freed up in 24.3.1).
"""

from mcp_abacus.expr.lexer import (
    COMMA,
    EOF,
    LPAREN,
    NAME,
    NEWLINE,
    NUMBER,
    OP,
    RPAREN,
    Token,
    tokenize,
)
from mcp_abacus.expr.nodes import (
    CONSTANT_NAMES,
    FUNCTION_ARITIES,
    Assign,
    BinOp,
    FuncCall,
    Node,
    Number,
    Sequence,
    UnaryOp,
    Var,
    _arity_ok,
    _describe_arity,
)

# Loose -> tight; the bitwise rungs sit below additive, ordered | < ^ < & (24.3.2).
_BINDING_POWER: dict[str, int] = {
    "|": 4,
    "^": 6,
    "&": 8,
    "+": 10,
    "-": 10,
    "*": 20,
    "/": 20,
    "//": 20,
    "%": 20,
}
_UNARY_OPS = frozenset({"+", "-", "~"})  # ~ is bitwise NOT (24.3.2)
_POWER_OPS = frozenset({"**"})


class ParseError(Exception):
    """Parse failure, carrying the 1-based input line of the offending token."""

    def __init__(self, message: str, line: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


def parse(text: str) -> Node:
    """Parse the input into an AST (20.2.2 / 30.5).

    One or more newline-separated statements (each an assignment or expression,
    30.3). A single statement returns its bare node; several return a Sequence
    wrapping them in order (30.5). Raises LexError (from tokenization) or
    ParseError, both carrying the 1-based input line.
    """
    return _Parser(tokenize(text)).program()


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0  # parenthesis nesting; NEWLINE is insignificant while > 0 (30.5)

    def _skip_continuations(self) -> None:
        # Inside parentheses a NEWLINE is just whitespace — it continues the
        # expression (the only way to span lines, 30.5). At statement level
        # (depth 0) NEWLINE stays visible, for program()/_binary to act on.
        if self._depth > 0:
            while self._tokens[self._pos].kind == NEWLINE:
                self._pos += 1

    def _peek(self) -> Token:
        self._skip_continuations()
        return self._tokens[self._pos]

    def _peek_at(self, offset: int) -> Token:
        # Bounded RAW lookahead (no continuation-skipping) used only by assignment's
        # NAME '=' peek at statement level; the list always ends with EOF, so an
        # offset past the end clamps to it rather than raising.
        pos = self._pos + offset
        return self._tokens[pos] if pos < len(self._tokens) else self._tokens[-1]

    def _advance(self) -> Token:
        self._skip_continuations()
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _match_op(self, ops: frozenset[str]) -> Token | None:
        token = self._peek()
        if token.kind == OP and token.lexeme in ops:
            return self._advance()
        return None

    def program(self) -> Node:
        """Parse the whole input: newline-separated statements (30.5).

        Each statement is an assignment or expression (30.3); statements are
        separated by one or more NEWLINEs (blank lines collapse). A lone statement
        returns its bare node; several wrap in a Sequence keyed to the first's line.
        A token that is neither NEWLINE nor EOF after a complete statement is
        trailing garbage (the same ``unexpected ... after the expression`` as a
        single bad expression).
        """
        self._skip_separators()  # leading blank lines
        if self._peek().kind == EOF:
            raise ParseError("empty input: expected an expression", self._peek().line)
        statements = [self.assignment()]
        while self._peek().kind != EOF:
            token = self._peek()
            if token.kind != NEWLINE:
                raise ParseError(f"unexpected {token.lexeme!r} after the expression", token.line)
            self._skip_separators()
            if self._peek().kind == EOF:  # trailing blank lines
                break
            statements.append(self.assignment())
        if len(statements) == 1:
            return statements[0]
        return Sequence(tuple(statements), line=statements[0].line)

    def _skip_separators(self) -> None:
        # Consume statement-separating NEWLINEs at depth 0 (peek/advance don't skip
        # them there). A no-op inside parentheses, where _peek never yields NEWLINE.
        while self._peek().kind == NEWLINE:
            self._advance()

    def assignment(self) -> Node:
        """Parse ``NAME '=' expr`` (an assignment) or fall through to an expression (30.3).

        Assignment is the LOOSEST precedence — it wraps the whole binary/unary/power
        chain. Recognised by a two-token lookahead (a NAME immediately followed by
        '='), so a bare NAME that is NOT an assignment target still descends into an
        ordinary expression (a call or a variable reference, 30.4). The right-hand
        side is a full expression; the Assign node keeps the name lexeme and that
        subtree. Only ``program`` calls this, once per statement — RHS / parens /
        arguments use ``expression``, so assignment never nests inside an expression
        (statement level only).
        """
        name = self._peek()
        after = self._peek_at(1)
        if name.kind == NAME and after.kind == OP and after.lexeme == "=":
            if name.lexeme in CONSTANT_NAMES:
                # pi/e are reserved constants (29.6) — binding one would shadow the
                # constant, so reject it at the target rather than silently rebind.
                raise ParseError(f"{name.lexeme!r} is a constant and cannot be assigned", name.line)
            self._advance()  # NAME
            self._advance()  # '='
            return Assign(name.lexeme, self.expression(), line=name.line)
        return self.expression()

    def expression(self) -> Node:
        return self._binary(1)

    def _binary(self, min_bp: int) -> Node:
        node = self._unary()
        while True:
            token = self._peek()
            if token.kind != OP:
                return node
            bp = _BINDING_POWER.get(token.lexeme)
            if bp is None or bp < min_bp:
                return node
            op = self._advance()
            # bp + 1 as the right-hand floor makes every table level left-assoc.
            node = BinOp(op.lexeme, node, self._binary(bp + 1), line=op.line)

    def _unary(self) -> Node:
        if (op := self._match_op(_UNARY_OPS)) is not None:
            return UnaryOp(op.lexeme, self._unary(), line=op.line)
        return self._power()

    def _power(self) -> Node:
        node = self._atom()
        if (op := self._match_op(_POWER_OPS)) is not None:
            # Right-assoc via the recursion; the exponent may be unary (2**-3).
            node = BinOp("**", node, self._unary(), line=op.line)
        return node

    def _atom(self) -> Node:
        token = self._advance()
        if token.kind == NUMBER:
            return Number(token.lexeme, line=token.line)
        if token.kind == NAME:
            # A NAME followed by '(' is a call / nullary (22.2 / 29.2); a bare NAME
            # is a variable reference (30.4). The two forms are kept distinct here.
            if self._peek().kind == LPAREN:
                return self._call(token)
            if token.lexeme in CONSTANT_NAMES:
                # Bare pi / e are reserved constants (29.6) — the same nullary call
                # as pi()/e(), so they evaluate identically without the parentheses.
                return FuncCall(token.lexeme, (), line=token.line)
            return Var(token.lexeme, line=token.line)
        if token.kind == LPAREN:
            # Inside the parens NEWLINE is insignificant — bracketing is the one way
            # an expression spans lines (30.5); the depth gates that in peek/advance.
            self._depth += 1
            node = self.expression()
            closing = self._advance()
            self._depth -= 1
            if closing.kind != RPAREN:
                raise ParseError(f"expected ')', got {_describe(closing)}", closing.line)
            return node  # parens only shape the tree — no Group node (20.2.2)
        raise ParseError(f"expected a number, name, or '(', got {_describe(token)}", token.line)

    def _call(self, name: Token) -> Node:
        """Parse ``NAME '(' (expr (',' expr)*)? ')'`` after the NAME (22.2 / 29.2).

        The argument list may be EMPTY — ``pi()`` is a nullary call (29.2). Whether
        zero arguments is allowed is left to the arity check below: a nullary name
        accepts it, any other name surfaces the precise wrong-count error. Name and
        arity are checked against the registry, the call's line being the NAME's.
        """
        opening = self._advance()
        if opening.kind != LPAREN:
            raise ParseError(
                f"expected '(' after function name {name.lexeme!r}, got {_describe(opening)}",
                name.line,
            )
        # The argument list brackets like parens — NEWLINE insignificant within (30.5).
        self._depth += 1
        args: list[Node] = []
        if self._peek().kind != RPAREN:
            args.append(self.expression())
            while self._peek().kind == COMMA:
                self._advance()
                args.append(self.expression())
        closing = self._advance()
        self._depth -= 1
        if closing.kind != RPAREN:
            raise ParseError(f"expected ',' or ')', got {_describe(closing)}", closing.line)
        if name.lexeme not in FUNCTION_ARITIES:
            raise ParseError(f"unknown function {name.lexeme!r}", name.line)
        if not _arity_ok(name.lexeme, len(args)):
            lo, hi = FUNCTION_ARITIES[name.lexeme]
            raise ParseError(
                f"function {name.lexeme!r} takes {_describe_arity(lo, hi)}, but {len(args)} given",
                name.line,
            )
        return FuncCall(name.lexeme, tuple(args), line=name.line)


def _describe(token: Token) -> str:
    if token.kind == EOF:
        return "end of input"
    if token.kind == NEWLINE:
        return "end of line"  # an incomplete expression cut off by a statement break (30.5)
    return repr(token.lexeme)
