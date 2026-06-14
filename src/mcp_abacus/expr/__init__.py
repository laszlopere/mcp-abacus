# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Expression engine subpackage: lexer, parser, AST nodes + in-node evaluation."""

from mcp_abacus.expr.lexer import LexError, Token, tokenize
from mcp_abacus.expr.nodes import (
    BINARY_OPS,
    UNARY_OPS,
    Assign,
    BinOp,
    EvalError,
    FuncCall,
    Node,
    Number,
    Sequence,
    UnaryOp,
    Var,
)
from mcp_abacus.expr.parser import ParseError, parse
from mcp_abacus.expr.value import (
    EvalContext,
    Mode,
    NotRepresentableError,
    UndefinedVariableError,
    Value,
    VariableStore,
)

__all__ = [
    "BINARY_OPS",
    "UNARY_OPS",
    "Assign",
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
    "Sequence",
    "Token",
    "UnaryOp",
    "UndefinedVariableError",
    "Value",
    "Var",
    "VariableStore",
    "parse",
    "tokenize",
]
