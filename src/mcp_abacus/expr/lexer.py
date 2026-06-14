# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

r"""Tokenizer for the expression language: source text -> tokens (TODO 20.1).

NUMBER lexemes are kept as RAW SOURCE text (17.2.2) and are UNSIGNED — sign is
parsed as a unary operator. Spaces/tabs are skipped; a shell-style '#' comment
runs to (but not through) the end of its line and is skipped the same way; a '\n'
is emitted as a NEWLINE token AND increments the line counter. NEWLINE is a
STATEMENT SEPARATOR
(30.5): the parser ends a statement at one, so an expression spans lines only
inside parentheses (where the parser treats NEWLINE as insignificant). Every
token carries the 1-based line it started on — this is what feeds node.line
(17.2.6).

Fixed-point DECIMALS notation (20.5). A fixed-point literal takes its scale from
one of two places: a DECIMAL's own digits (1.5 -> scale 1, 1.50 -> scale 2;
trailing zeros are significant, cf. Decimal("1.50")), or a BASE-PREFIXED RAW
INTEGER (hex/octal/binary) plus an explicit '@'<decimals> tag — for on-chain
amounts (ERC-20 / Uniswap reserves arrive as a raw uint with the decimals known
out-of-band). '@' attaches ONLY to base-prefixed integers; a plain decimal sets
its scale by writing the digits (123.45), never with '@' (19.2.1):

    <base-int> '@' <decimals>   ==   M x 10^-D
    0x59682F00@9         = 1.5
    0xDE0B6B3A7640000@18 = 1.0 ETH

'@' is chosen over ':' / 'p' / '#' because it is not a hex digit (so it is clean
right after 0x...), is not an existing operator, and reads "at N decimals".
Scientific 'e' CANNOT serve here: 'e' is a hex digit, so 0x..e9 would be
ambiguous. The lexer only carries the '@'<decimals> form as RAW SOURCE; turning
D into the decimal count is a Value concern (M x 10^-D in every mode, 20.5.2),
and how DIFFERING scales combine is fixed-point op semantics (10.1.1) — neither
is a notation question.
"""

import re
from dataclasses import dataclass
from typing import NoReturn

NUMBER = "NUMBER"
NAME = "NAME"  # identifier: function names (22.1); the function set is not the lexer's concern
OP = "OP"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
COMMA = "COMMA"  # argument separator, reserved for n-ary function calls (22.1)
NEWLINE = "NEWLINE"  # statement separator (30.5); insignificant inside parentheses
EOF = "EOF"

TOKEN_KINDS: frozenset[str] = frozenset({NUMBER, NAME, OP, LPAREN, RPAREN, COMMA, NEWLINE, EOF})

_DIGITS = frozenset("0123456789")
# ^ & | bitwise, ~ bitwise NOT (24.3.2); = is the assignment operator (30.3) — not a
# value operator, so the parser only ever consumes it at statement level, never in _binary.
_SINGLE_CHAR_OPS = frozenset("+-*/%^&|~=")
_BASE_PREFIXES = ("0x", "0X", "0b", "0B", "0o", "0O")
# ASCII-only on purpose: no unicode digits, no underscore separators (20.1.3).
_BASE_INTEGER = re.compile(r"0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+")
_DECIMAL = re.compile(r"(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INT_DIGITS = re.compile(r"[0-9]+")  # a plain integer body; also the '@'<decimals> tail
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")  # ASCII identifier (22.1)
_NUMBERISH = re.compile(r"[\w.]*")  # only for malformed-number error messages


class LexError(Exception):
    """Tokenization failure, carrying the 1-based input line."""

    def __init__(self, message: str, line: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


@dataclass(frozen=True, slots=True)
class Token:
    """One lexical token; lexeme is the raw source text (empty for EOF)."""

    kind: str
    lexeme: str
    line: int

    def __post_init__(self) -> None:
        if self.kind not in TOKEN_KINDS:
            raise ValueError(f"unknown token kind: {self.kind!r}")
        if self.line < 1:
            raise ValueError(f"line must be >= 1, got {self.line}")


def tokenize(text: str) -> list[Token]:
    """Tokenize the whole input; the result always ends with an EOF token."""
    tokens: list[Token] = []
    line = 1
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            tokens.append(Token(NEWLINE, "\n", line))  # statement separator (30.5)
            line += 1
            i += 1
        elif char == "#":
            # Shell-style comment: '#' to end of line is skipped like whitespace. The
            # terminating '\n' is left for the next iteration so it still becomes a
            # NEWLINE statement separator (30.5); a '#' on the last line runs to EOF.
            newline = text.find("\n", i)
            i = len(text) if newline == -1 else newline
        elif char.isspace():
            i += 1
        elif char in _DIGITS or (char == "." and text[i + 1 : i + 2] in _DIGITS):
            lexeme, i = _lex_number(text, i, line)
            tokens.append(Token(NUMBER, lexeme, line))
        elif (match := _NAME.match(text, i)) is not None:
            tokens.append(Token(NAME, match.group(), line))
            i = match.end()
        elif text.startswith("**", i):  # longest match: power, never two '*'
            tokens.append(Token(OP, "**", line))
            i += 2
        elif text.startswith("//", i):  # longest match: ONE token, never two '/'
            tokens.append(Token(OP, "//", line))
            i += 2
        elif char in _SINGLE_CHAR_OPS:
            tokens.append(Token(OP, char, line))
            i += 1
        elif char == "(":
            tokens.append(Token(LPAREN, char, line))
            i += 1
        elif char == ")":
            tokens.append(Token(RPAREN, char, line))
            i += 1
        elif char == ",":
            tokens.append(Token(COMMA, char, line))
            i += 1
        else:
            raise LexError(f"unexpected character: {char!r}", line)
    tokens.append(Token(EOF, "", line))
    return tokens


def _lex_number(text: str, start: int, line: int) -> tuple[str, int]:
    """Lex one number literal at text[start:]; return (raw lexeme, end index).

    An INTEGER literal (any base) may carry a '@'<decimals> suffix (20.5.1); the
    whole thing is kept as one raw NUMBER lexeme, so it binds tighter than every
    operator (0xFF@9**2 == (0xFF@9)**2).
    """
    if text.startswith(_BASE_PREFIXES, start):
        match = _BASE_INTEGER.match(text, start)  # base-prefixed lexemes are integers
        is_base_prefixed = True
    else:
        # The caller guarantees a digit or '.'-digit start, so this matches.
        match = _DECIMAL.match(text, start)
        is_base_prefixed = False
    if match is None:
        _malformed(text, start, start, line)
    end = match.end()
    if text[end : end + 1] == "@":
        return _lex_at_decimals(text, start, end, line, is_base_prefixed)
    return _finish(text, start, end, line)


def _lex_at_decimals(
    text: str, start: int, at: int, line: int, is_base_prefixed: bool
) -> tuple[str, int]:
    """Lex the '@'<decimals> suffix; `at` indexes the '@' (20.5.1).

    The suffix attaches only to base-prefixed (hex/octal/binary) integers — a raw
    integer that has no other way to write a fraction. A decimal literal already
    sets its scale by writing its own digits (67 stays 67, 67.00 is scale 2), so
    '@' on a decimal — integer or not — is rejected (19.2.1). <decimals> is one or
    more plain decimal digits, '@0' being the bare integer. Interpreting D as the
    decimal count happens in Value (20.5.2).
    """
    if not is_base_prefixed:
        raise LexError(
            f"'@' decimals suffix attaches only to base-prefixed (hex/octal/binary) "
            f"integers, not decimal {text[start:at]!r}",
            line,
        )
    digits = _INT_DIGITS.match(text, at + 1)
    if digits is None:
        raise LexError(f"'@' must be followed by decimal digits: {text[start : at + 1]!r}", line)
    return _finish(text, start, digits.end(), line)


def _finish(text: str, start: int, end: int, line: int) -> tuple[str, int]:
    """Accept text[start:end] as a lexeme if a clean boundary follows, else raise."""
    following = text[end : end + 1]
    if following != "." and not following.isalnum():
        return text[start:end], end
    _malformed(text, start, end, line)


def _malformed(text: str, start: int, end: int, line: int) -> NoReturn:
    """Raise a malformed-number LexError, naming the offending run of source."""
    tail = _NUMBERISH.match(text, end)
    assert tail is not None  # a '*' pattern always matches (possibly empty)
    bad = text[start:end] + tail.group()
    raise LexError(f"malformed number: {bad!r}", line)
