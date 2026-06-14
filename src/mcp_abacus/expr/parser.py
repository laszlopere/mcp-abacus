# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Recursive-descent parser: source text -> AST nodes (TODO 20.2).

Precedence, loose -> tight (20.2.1 — math convention, NOT Excel; the bitwise
rungs and ~ follow C/Python, 24.3.2):
bitwise OR ``|`` < bitwise XOR ``^`` < bitwise AND ``&`` < additive ``+ -``
< multiplicative ``* / // %`` (all left-assoc) < unary ``+ - ~`` < power ``**``
(RIGHT-assoc and tighter than unary minus: ``-2**2 == -(2**2)``; the exponent may
itself be unary: ``2**-3``). Atoms are numbers, parenthesized expressions, and
function calls ``NAME '(' expr (',' expr)* ')'`` (22.2 — tightest binding, so
``sqrt(4)**2`` is ``(sqrt(4))**2``); the call's name and arity are validated here
against nodes.FUNCTION_ARITIES, unknown-name/wrong-arity being parse errors.
Parens only shape the tree — no Group node. The left-assoc binary levels run on a
Pratt binding-power table; unary/power/atom are plain descent.

``^`` is bitwise XOR, NOT power (power is ``**``, freed up in 24.3.1).
"""

from mcp_abacus.expr.lexer import COMMA, EOF, LPAREN, NAME, NUMBER, OP, RPAREN, Token, tokenize
from mcp_abacus.expr.nodes import (
    FUNCTION_ARITIES,
    BinOp,
    FuncCall,
    Node,
    Number,
    UnaryOp,
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
    """Parse exactly ONE expression (which may span lines) into an AST (20.2.2).

    Raises LexError (from tokenization) or ParseError, both carrying the
    1-based input line.
    """
    tokens = tokenize(text)
    if tokens[0].kind == EOF:
        raise ParseError("empty input: expected an expression", tokens[0].line)
    parser = _Parser(tokens)
    node = parser.expression()
    parser.expect_end()
    return node


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _match_op(self, ops: frozenset[str]) -> Token | None:
        token = self._peek()
        if token.kind == OP and token.lexeme in ops:
            return self._advance()
        return None

    def expression(self) -> Node:
        return self._binary(1)

    def expect_end(self) -> None:
        token = self._peek()
        if token.kind != EOF:
            raise ParseError(f"unexpected {token.lexeme!r} after the expression", token.line)

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
            return self._call(token)
        if token.kind == LPAREN:
            node = self.expression()
            closing = self._advance()
            if closing.kind != RPAREN:
                raise ParseError(f"expected ')', got {_describe(closing)}", closing.line)
            return node  # parens only shape the tree — no Group node (20.2.2)
        raise ParseError(
            f"expected a number, function name, or '(', got {_describe(token)}", token.line
        )

    def _call(self, name: Token) -> Node:
        """Parse ``NAME '(' expr (',' expr)* ')'`` after the NAME (22.2).

        At least one argument — there are no zero-arg functions yet, so an empty
        ``()`` surfaces as the atom parser's "expected a number..." error. Name and
        arity are checked against the registry, the call's line being the NAME's.
        """
        opening = self._advance()
        if opening.kind != LPAREN:
            raise ParseError(
                f"expected '(' after function name {name.lexeme!r}, got {_describe(opening)}",
                name.line,
            )
        args = [self.expression()]
        while self._peek().kind == COMMA:
            self._advance()
            args.append(self.expression())
        closing = self._advance()
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
    return "end of input" if token.kind == EOF else repr(token.lexeme)
