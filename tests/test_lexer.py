# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the expression tokenizer (TODO 20.3)."""

import dataclasses

import pytest

from mcp_abacus.expr.lexer import (
    COMMA,
    EOF,
    LPAREN,
    NAME,
    NUMBER,
    OP,
    RPAREN,
    LexError,
    Token,
    tokenize,
)


def _kinds_and_lexemes(tokens: list[Token]) -> list[tuple[str, str]]:
    return [(token.kind, token.lexeme) for token in tokens]


def test_simple_expression():
    tokens = tokenize("1 + 2 * 10**3")
    assert _kinds_and_lexemes(tokens) == [
        (NUMBER, "1"),
        (OP, "+"),
        (NUMBER, "2"),
        (OP, "*"),
        (NUMBER, "10"),
        (OP, "**"),
        (NUMBER, "3"),
        (EOF, ""),
    ]
    assert all(token.line == 1 for token in tokens)


def test_all_operators_and_parens():
    tokens = tokenize("(1+2)-3*4/5//6%7**8")
    assert _kinds_and_lexemes(tokens) == [
        (LPAREN, "("),
        (NUMBER, "1"),
        (OP, "+"),
        (NUMBER, "2"),
        (RPAREN, ")"),
        (OP, "-"),
        (NUMBER, "3"),
        (OP, "*"),
        (NUMBER, "4"),
        (OP, "/"),
        (NUMBER, "5"),
        (OP, "//"),
        (NUMBER, "6"),
        (OP, "%"),
        (NUMBER, "7"),
        (OP, "**"),
        (NUMBER, "8"),
        (EOF, ""),
    ]


def test_floordiv_is_one_token():
    assert _kinds_and_lexemes(tokenize("1//2")) == [
        (NUMBER, "1"),
        (OP, "//"),
        (NUMBER, "2"),
        (EOF, ""),
    ]


def test_separated_slashes_stay_two_tokens():
    assert _kinds_and_lexemes(tokenize("1/ /2")) == [
        (NUMBER, "1"),
        (OP, "/"),
        (OP, "/"),
        (NUMBER, "2"),
        (EOF, ""),
    ]


def test_double_star_is_one_token():
    assert _kinds_and_lexemes(tokenize("2**3")) == [
        (NUMBER, "2"),
        (OP, "**"),
        (NUMBER, "3"),
        (EOF, ""),
    ]


def test_separated_stars_stay_two_tokens():
    assert _kinds_and_lexemes(tokenize("2* *3")) == [
        (NUMBER, "2"),
        (OP, "*"),
        (OP, "*"),
        (NUMBER, "3"),
        (EOF, ""),
    ]


def test_caret_lexes_as_op():
    # '^' is the bitwise XOR operator (24.3.2); the lexer emits it as a plain OP.
    assert _kinds_and_lexemes(tokenize("5^3")) == [
        (NUMBER, "5"),
        (OP, "^"),
        (NUMBER, "3"),
        (EOF, ""),
    ]


def test_bitwise_ops_lex_as_single_char_ops():
    # & | ~ are the remaining bitwise operators (24.3.2), each a single-char OP.
    assert _kinds_and_lexemes(tokenize("1&2|~3")) == [
        (NUMBER, "1"),
        (OP, "&"),
        (NUMBER, "2"),
        (OP, "|"),
        (OP, "~"),
        (NUMBER, "3"),
        (EOF, ""),
    ]


@pytest.mark.parametrize(
    "lexeme",
    [
        "42",
        "0",
        "0.1",
        "3.",
        ".5",
        "1e-3",
        "2.5E+6",
        "7e2",
        "0xFF",
        "0Xff",
        "0b1010",
        "0B1",
        "0o17",
        "0O17",
    ],
)
def test_number_lexemes_kept_as_raw_source(lexeme):
    assert _kinds_and_lexemes(tokenize(lexeme)) == [(NUMBER, lexeme), (EOF, "")]


@pytest.mark.parametrize(
    "lexeme",
    ["0x3039@2", "0x2A@0", "0x59682F00@9", "0xDE0B6B3A7640000@18", "0b1010@4", "0o17@3"],
)
def test_at_decimals_suffix_kept_as_raw_source(lexeme):
    assert _kinds_and_lexemes(tokenize(lexeme)) == [(NUMBER, lexeme), (EOF, "")]


def test_at_decimals_binds_to_the_atom():
    # 0xFF@9 is ONE lexeme, so '**' applies to the whole tag: (0xFF@9)**2.
    assert _kinds_and_lexemes(tokenize("0xFF@9**2")) == [
        (NUMBER, "0xFF@9"),
        (OP, "**"),
        (NUMBER, "2"),
        (EOF, ""),
    ]


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("1.5@9", 1),  # '@' attaches only to base-prefixed integers
        ("1\n1.5@9", 2),
        ("1e3@9", 1),  # scientific already fixes the scale
        ("12345@2", 1),  # decimal integer: '@' rejected (19.2.1)
        ("67@8", 1),
        ("1\n67@8", 2),
        ("12@", 1),  # trailing '@' with no digits
        ("1\n12@", 2),
        ("0xFF@", 1),
    ],
)
def test_invalid_at_suffix_raises_with_line(text, line):
    with pytest.raises(LexError) as excinfo:
        tokenize(text)
    assert excinfo.value.line == line


def test_function_call_shape_lexes_with_name_and_comma():
    assert _kinds_and_lexemes(tokenize("max(1, 2)")) == [
        (NAME, "max"),
        (LPAREN, "("),
        (NUMBER, "1"),
        (COMMA, ","),
        (NUMBER, "2"),
        (RPAREN, ")"),
        (EOF, ""),
    ]


@pytest.mark.parametrize("lexeme", ["sqrt", "abs", "_x", "log2", "PI", "a1b2"])
def test_name_lexemes(lexeme):
    assert _kinds_and_lexemes(tokenize(lexeme)) == [(NAME, lexeme), (EOF, "")]


def test_name_stops_at_non_identifier_char():
    # A name is greedy over [A-Za-z0-9_] and stops at the operator (no whitespace needed).
    assert _kinds_and_lexemes(tokenize("abs+1")) == [
        (NAME, "abs"),
        (OP, "+"),
        (NUMBER, "1"),
        (EOF, ""),
    ]


def test_number_followed_by_name_is_malformed():
    # A digit-led run can't become a name; '2x' is a single malformed number, not NUMBER+NAME.
    with pytest.raises(LexError, match="malformed number"):
        tokenize("2x")


def test_multiline_input_line_numbers():
    tokens = tokenize("1 +\n2 *\n\n3")
    assert [(token.kind, token.lexeme, token.line) for token in tokens] == [
        (NUMBER, "1", 1),
        (OP, "+", 1),
        (NUMBER, "2", 2),
        (OP, "*", 2),
        (NUMBER, "3", 4),
        (EOF, "", 4),
    ]


def test_empty_and_whitespace_only_input():
    assert _kinds_and_lexemes(tokenize("")) == [(EOF, "")]
    tokens = tokenize(" \t\n ")
    assert _kinds_and_lexemes(tokens) == [(EOF, "")]
    assert tokens[0].line == 2


@pytest.mark.parametrize(
    "text",
    ["0x", "0b", "0o", "0xZZ", "0b12", "0o19", "0x.5", "1e", "1e+", "1.2.3", "1..2", ".5.6"],
)
def test_malformed_numbers_raise(text):
    with pytest.raises(LexError, match="malformed number"):
        tokenize(text)


def test_unknown_character_raises():
    with pytest.raises(LexError) as excinfo:
        tokenize("1 + $2")
    assert excinfo.value.line == 1


def test_bare_dot_raises():
    with pytest.raises(LexError):
        tokenize("3 . 5")


def test_lexerror_carries_correct_line():
    with pytest.raises(LexError) as excinfo:
        tokenize("1 +\n2 +\n0x")
    assert excinfo.value.line == 3
    with pytest.raises(LexError) as excinfo:
        tokenize("1\n@")
    assert excinfo.value.line == 2


def test_token_validation():
    token = Token(NUMBER, "1", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        token.lexeme = "2"  # type: ignore[misc]
    with pytest.raises(ValueError):
        Token("BOGUS", "1", 1)
    with pytest.raises(ValueError):
        Token(NUMBER, "1", 0)
