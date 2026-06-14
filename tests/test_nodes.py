# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the expression AST node classes (TODO 17.3)."""

import dataclasses

import pytest

from mcp_abacus.expr.nodes import (
    BINARY_OPS,
    FUNCTION_ARITIES,
    UNARY_OPS,
    BinOp,
    FuncCall,
    Node,
    Number,
    UnaryOp,
)


def _example_tree(line: int = 1) -> BinOp:
    """The CORE CONCEPT example "1 + 2 * 10**3", every node at the given line."""
    return BinOp(
        "+",
        Number("1", line=line),
        BinOp(
            "*",
            Number("2", line=line),
            BinOp("**", Number("10", line=line), Number("3", line=line), line=line),
            line=line,
        ),
        line=line,
    )


def test_number_round_trip():
    n = Number("0.1", line=3)
    assert n.lexeme == "0.1"
    assert n.line == 3
    assert Number("1e-3", line=1).lexeme == "1e-3"


def test_unaryop_and_binop_round_trip():
    operand = Number("5", line=2)
    u = UnaryOp("-", operand, line=2)
    assert u.op == "-"
    assert u.operand is operand

    left = Number("1", line=4)
    right = Number("2", line=4)
    b = BinOp("//", left, right, line=4)
    assert b.op == "//"
    assert b.left is left
    assert b.right is right
    assert b.line == 4


def test_example_tree_nests_correctly():
    tree = _example_tree()
    assert tree.op == "+"
    assert isinstance(tree.left, Number) and tree.left.lexeme == "1"
    mul = tree.right
    assert isinstance(mul, BinOp) and mul.op == "*"
    assert isinstance(mul.left, Number) and mul.left.lexeme == "2"
    power = mul.right
    assert isinstance(power, BinOp) and power.op == "**"
    assert isinstance(power.left, Number) and power.left.lexeme == "10"
    assert isinstance(power.right, Number) and power.right.lexeme == "3"


def test_structural_equality_ignores_line():
    assert _example_tree(1) == _example_tree(7)
    assert Number("1", line=1) != Number("2", line=1)


def test_hashable_and_hash_ignores_line():
    assert hash(_example_tree(1)) == hash(_example_tree(7))
    assert len({_example_tree(1), _example_tree(7)}) == 1


def test_frozen():
    n = Number("1", line=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.lexeme = "2"  # type: ignore[misc]
    b = BinOp("+", Number("1", line=1), Number("2", line=1), line=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.op = "-"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.line = 2  # type: ignore[misc]


def test_unknown_op_raises():
    one = Number("1", line=1)
    with pytest.raises(ValueError):
        UnaryOp("*", one, line=1)
    with pytest.raises(ValueError):
        BinOp("~", one, one, line=1)  # ~ is unary NOT, not a binary op
    with pytest.raises(ValueError):
        BinOp("<<", one, one, line=1)  # shifts are not in the language (yet)


def test_funccall_round_trip():
    operand = Number("4", line=2)
    call = FuncCall("sqrt", (operand,), line=2)
    assert call.name == "sqrt"
    assert call.args == (operand,)
    assert call.line == 2


def test_funccall_unknown_name_raises():
    with pytest.raises(ValueError):
        FuncCall("nope", (Number("1", line=1),), line=1)


def test_funccall_wrong_arity_raises():
    one = Number("1", line=1)
    with pytest.raises(ValueError):
        FuncCall("sqrt", (), line=1)  # sqrt is unary; zero args is wrong
    with pytest.raises(ValueError):
        FuncCall("sqrt", (one, one), line=1)  # two args is wrong


def test_funccall_variadic_arity():
    one = Number("1", line=1)
    # sum is variadic (>= 1 arg): zero is too few, one-or-more all construct.
    with pytest.raises(ValueError):
        FuncCall("sum", (), line=1)
    assert FuncCall("sum", (one,), line=1).args == (one,)
    assert FuncCall("sum", (one, one, one), line=1).args == (one, one, one)


def test_function_arities_match_the_registry():
    # FUNCTION_ARITIES is derived from _FUNCS — every wired function carries an
    # arity range (min, max|None): unary funcs are (1, 1), binary pow (2, 2),
    # variadic sum (1, None).
    assert FUNCTION_ARITIES == {
        "abs": (1, 1),
        "sqrt": (1, 1),
        "pow": (2, 2),  # the only BINARY function — fixed arity 2 (28.20)
        "sin": (1, 1),
        "cos": (1, 1),
        "tan": (1, 1),
        "cot": (1, 1),
        "log": (1, 1),
        "ln": (1, 1),
        "log10": (1, 1),
        "sum": (1, None),
        "product": (1, None),
        "avg": (1, None),
        "max": (1, None),
        "min": (1, None),
        "median": (1, None),
        "variance": (1, None),
        "stddev": (1, None),
    }


def test_empty_lexeme_raises():
    with pytest.raises(ValueError):
        Number("", line=1)


def test_line_below_one_raises():
    one = Number("1", line=1)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            Number("1", line=bad)
        with pytest.raises(ValueError):
            UnaryOp("-", one, line=bad)
        with pytest.raises(ValueError):
            BinOp("+", one, one, line=bad)
        with pytest.raises(ValueError):
            FuncCall("sqrt", (one,), line=bad)


def test_node_base_is_abstract():
    with pytest.raises(TypeError):
        Node()  # type: ignore[abstract]


def test_match_case_destructuring():
    match _example_tree():
        case BinOp("+", Number(a), BinOp("*", Number(b), BinOp("**", Number(c), Number(d)))):
            assert (a, b, c, d) == ("1", "2", "10", "3")
        case _:
            pytest.fail("example tree did not match the expected pattern")


def test_pretty_golden():
    # Unevaluated tree: kind + op/quoted-lexeme only, no line numbers (26.2/26.4).
    expected = (
        "BINARY_ADD\n"
        '  LITERAL "1"\n'
        "  BINARY_MUL\n"
        '    LITERAL "2"\n'
        "    BINARY_POW\n"
        '      LITERAL "10"\n'
        '      LITERAL "3"'
    )
    assert _example_tree().pretty() == expected


def test_op_sets():
    assert UNARY_OPS == frozenset({"+", "-", "~"})
    assert BINARY_OPS == frozenset({"+", "-", "*", "/", "//", "%", "**", "&", "|", "^"})
    assert isinstance(UNARY_OPS, frozenset)
    assert isinstance(BINARY_OPS, frozenset)


def test_reexports_from_expr_package():
    import mcp_abacus.expr as expr
    import mcp_abacus.expr.nodes as nodes

    for name in ("BINARY_OPS", "UNARY_OPS", "BinOp", "FuncCall", "Node", "Number", "UnaryOp"):
        assert getattr(expr, name) is getattr(nodes, name)
