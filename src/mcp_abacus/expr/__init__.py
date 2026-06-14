# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Expression engine subpackage: lexer, parser, AST nodes + in-node evaluation."""

from mcp_abacus.expr.lexer import LexError, Token, tokenize
from mcp_abacus.expr.nodes import (
    BINARY_OPS,
    UNARY_OPS,
    BinOp,
    EvalError,
    FuncCall,
    Node,
    Number,
    UnaryOp,
)
from mcp_abacus.expr.parser import ParseError, parse
from mcp_abacus.expr.value import EvalContext, Mode, NotRepresentableError, Value

__all__ = [
    "BINARY_OPS",
    "UNARY_OPS",
    "BinOp",
    "EvalContext",
    "EvalError",
    "FuncCall",
    "LexError",
    "Mode",
    "Node",
    "NotRepresentableError",
    "Number",
    "ParseError",
    "Token",
    "UnaryOp",
    "Value",
    "parse",
    "tokenize",
]
