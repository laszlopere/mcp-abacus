# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit tests for the VECTOR type and vector literals (TODO 19.1.10).

In-process tests of the four layers a vector touches: the lexer's bracket tokens,
the parser's ``[a, b, …]`` literal, the ``Vector`` payload + ``Value.vector``
constructor on the universal Value class, and the internal-only mode carve-out.
The over-the-wire behaviour lives in test_vectors_e2e.py; kept separate from the
existing per-layer suites so the vector cases stay in one place.
"""

import pytest

from mcp_abacus.expr.lexer import LBRACKET, RBRACKET, tokenize
from mcp_abacus.expr.nodes import Assign, Number, VectorLiteral
from mcp_abacus.expr.parser import ParseError, parse
from mcp_abacus.expr.value import (
    Mode,
    NotRepresentableError,
    Value,
    Vector,
    resolve_mode,
    selectable_modes,
)


def _f64(lexeme):
    return Value.from_lexeme(lexeme, Mode.FLOATING_POINT)


def _fp(lexeme):
    return Value.from_lexeme(lexeme, Mode.FIXED_POINT)


# --- lexer (20.1) ----------------------------------------------------------------


def test_lexer_emits_bracket_tokens():
    kinds = [(t.kind, t.lexeme) for t in tokenize("[1, 2]") if t.kind not in ("NEWLINE", "EOF")]
    assert kinds[0] == (LBRACKET, "[")
    assert kinds[-1] == (RBRACKET, "]")


# --- parser (20.2) ---------------------------------------------------------------


def test_parser_builds_vector_literal():
    node = parse("[1, 2, 3]")
    assert isinstance(node, VectorLiteral)
    assert len(node.elements) == 3
    assert all(isinstance(e, Number) for e in node.elements)
    assert node.source() == "[1, 2, 3]"


def test_parser_allows_empty_vector():
    node = parse("[]")
    assert isinstance(node, VectorLiteral)
    assert node.elements == ()
    assert node.source() == "[]"


def test_parser_vector_elements_are_full_expressions():
    node = parse("[1 + 2, 3 * 4]")
    assert isinstance(node, VectorLiteral)
    assert len(node.elements) == 2
    assert node.source() == "[1 + 2, 3 * 4]"


def test_parser_unclosed_vector_is_a_parse_error():
    with pytest.raises(ParseError, match=r"expected ',' or '\]'"):
        parse("[1, 2")


def test_parser_trailing_comma_is_a_parse_error():
    with pytest.raises(ParseError):
        parse("[1, 2,]")


# --- vectors in assignments (30.3) ----------------------------------------------
# A vector is a first-class Value, so a NAME = expr binding holds one with no special
# casing: Assign binds the RHS Value, Var reads it back, both generic over the type.


def test_parser_builds_a_vector_assignment():
    node = parse("v = [1.00, 1.50, 2.00]")
    assert isinstance(node, Assign)
    assert node.name == "v"
    assert isinstance(node.expr, VectorLiteral)
    assert len(node.expr.elements) == 3


def test_vector_assignment_binds_and_reads_back():
    # The program's value is the last statement's — here the bare reference, which
    # reads the vector bound on the first line straight out of the run's store.
    result = parse("v = [1.00, 1.50, 2.00]\nv").evaluate(Mode.FIXED_POINT)
    assert result.mode is Mode.VECTOR
    assert result.to_string() == "[1.00, 1.50, 2.00]"
    assert result.exact is True


def test_vector_assignment_chains_through_another_name():
    result = parse("a = [1, 2, 3]\nb = a\nb").evaluate(Mode.FIXED_POINT)
    assert result.to_string() == "[1, 2, 3]"


# --- Value.vector constructor + Vector payload (19.1.10) -------------------------


def test_value_vector_constructs_and_renders():
    v = Value.vector([_fp("1"), _fp("2"), _fp("3")], Mode.FIXED_POINT)
    assert v.mode is Mode.VECTOR
    assert isinstance(v.payload, Vector)
    assert v.payload.element_mode is Mode.FIXED_POINT
    assert v.to_string() == "[1, 2, 3]"
    assert v.precision() is None  # no single scale
    assert v.hex_dump() is None  # no single integer to dump


def test_value_vector_exactness_folds_elements():
    # Exact iff EVERY element is exact; an empty vector is vacuously exact.
    assert Value.vector([_fp("1"), _fp("2")], Mode.FIXED_POINT).exact is True
    assert Value.vector([], Mode.FIXED_POINT).exact is True
    inexact = _fp("1").div(_fp("3"))  # 1/3 rounds at scale 0 -> inexact
    assert inexact.exact is False
    assert Value.vector([_fp("1"), inexact], Mode.FIXED_POINT).exact is False


def test_value_vector_rejects_a_nested_vector():
    inner = Value.vector([_fp("1"), _fp("2")], Mode.FIXED_POINT)
    with pytest.raises(NotRepresentableError, match="one-dimensional"):
        Value.vector([inner], Mode.VECTOR)


def test_vector_payload_rejects_mode_mismatched_elements():
    # The structural invariant: every element shares element_mode.
    with pytest.raises(ValueError, match="element mode"):
        Vector(Mode.FIXED_POINT, (_f64("1.0"),))


def test_vector_describe_envelope():
    v = Value.vector([_fp("1"), _fp("2")], Mode.FIXED_POINT)
    assert v.describe() == "[1, 2] (vector, exact)"


# --- operator refusals at the Value layer (19.1.10) -----------------------------


def test_value_vectors_refuse_binary_and_unary_ops():
    a = Value.vector([_fp("1"), _fp("2")], Mode.FIXED_POINT)
    b = Value.vector([_fp("3"), _fp("4")], Mode.FIXED_POINT)
    with pytest.raises(NotRepresentableError, match=r"vectors do not support \+"):
        a.add(b)
    with pytest.raises(NotRepresentableError, match="vectors do not support unary -"):
        a.neg()


# --- internal-only mode carve-out -----------------------------------------------


def test_vector_is_not_selectable():
    assert Mode.VECTOR not in selectable_modes()
    assert all(m is not Mode.VECTOR for m in selectable_modes())
    with pytest.raises(ValueError):
        resolve_mode("vector")
