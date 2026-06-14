# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the recursive-descent expression parser (TODO 20.3)."""

from fractions import Fraction

import pytest

from mcp_abacus.expr.lexer import LexError
from mcp_abacus.expr.nodes import BinOp, FuncCall, Number, UnaryOp
from mcp_abacus.expr.parser import ParseError, parse
from mcp_abacus.expr.value import FixedPoint, Mode


def test_single_number():
    assert parse("42") == Number("42", line=1)


def test_core_concept_example_tree():
    expected = BinOp(
        "+",
        Number("1", line=1),
        BinOp(
            "*",
            Number("2", line=1),
            BinOp("**", Number("10", line=1), Number("3", line=1), line=1),
            line=1,
        ),
        line=1,
    )
    assert parse("1 + 2 * 10**3") == expected


def test_pretty_golden_mul_binds_tighter_than_add():
    assert parse("1+2*3").pretty() == (
        'BINARY_ADD\n  LITERAL "1"\n  BINARY_MUL\n    LITERAL "2"\n    LITERAL "3"'
    )


def test_pretty_golden_power_tighter_than_unary_minus():
    assert parse("-2**2").pretty() == ('UNARY_NEG\n  BINARY_POW\n    LITERAL "2"\n    LITERAL "2"')


def test_pretty_golden_power_right_assoc():
    assert parse("2**3**2").pretty() == (
        'BINARY_POW\n  LITERAL "2"\n  BINARY_POW\n    LITERAL "3"\n    LITERAL "2"'
    )


def test_pretty_golden_unary_exponent():
    assert parse("2**-3").pretty() == ('BINARY_POW\n  LITERAL "2"\n  UNARY_NEG\n    LITERAL "3"')


def test_caret_is_bitwise_xor():
    # ^ is XOR now (power is **, 24.3.1/24.3.2), so it parses as a plain binary op.
    assert parse("2^3") == BinOp("^", Number("2", line=1), Number("3", line=1), line=1)


def test_pretty_golden_bitwise_precedence_or_xor_and_below_additive():
    # loose -> tight: | < ^ < & < + (24.3.2), so each binds looser than the next.
    assert parse("1 | 2 ^ 3 & 4 + 5").pretty() == (
        "BINARY_OR\n"
        '  LITERAL "1"\n'
        "  BINARY_XOR\n"
        '    LITERAL "2"\n'
        "    BINARY_AND\n"
        '      LITERAL "3"\n'
        "      BINARY_ADD\n"
        '        LITERAL "4"\n'
        '        LITERAL "5"'
    )


def test_pretty_golden_bitwise_left_assoc():
    assert parse("1 ^ 2 ^ 3").pretty() == (
        'BINARY_XOR\n  BINARY_XOR\n    LITERAL "1"\n    LITERAL "2"\n  LITERAL "3"'
    )


def test_pretty_golden_bitnot_binds_tighter_than_and():
    # ~ is a unary prefix (tighter than the binary bitwise rungs): ~1 & 2 == (~1) & 2.
    assert parse("~1 & 2").pretty() == ('BINARY_AND\n  UNARY_NOT\n    LITERAL "1"\n  LITERAL "2"')


def test_pretty_golden_parens_shape_the_tree():
    assert parse("(1+2)*3").pretty() == (
        'BINARY_MUL\n  BINARY_ADD\n    LITERAL "1"\n    LITERAL "2"\n  LITERAL "3"'
    )


def test_pretty_golden_floordiv_mod_left_assoc():
    assert parse("7//2%3").pretty() == (
        'BINARY_MOD\n  BINARY_FLOORDIV\n    LITERAL "7"\n    LITERAL "2"\n  LITERAL "3"'
    )


def test_additive_left_assoc():
    assert parse("1-2+3").pretty() == (
        'BINARY_ADD\n  BINARY_SUB\n    LITERAL "1"\n    LITERAL "2"\n  LITERAL "3"'
    )


def test_pretty_golden_at_decimals_binds_tighter_than_power():
    # '@9' rides on the atom, so '**' sees the whole tag: (0xFF@9)**2.
    assert parse("0xFF@9**2").pretty() == ('BINARY_POW\n  LITERAL "0xFF@9"\n  LITERAL "2"')


def test_double_unary_minus():
    assert parse("--2") == UnaryOp("-", UnaryOp("-", Number("2", line=1), line=1), line=1)


def test_unary_plus():
    assert parse("+5") == UnaryOp("+", Number("5", line=1), line=1)


def test_redundant_parens_leave_no_trace():
    assert parse("((42))") == Number("42", line=1)


def test_node_lines_come_from_defining_tokens():
    # Lines are no longer shown in pretty() (26.4) but remain a node property that
    # feeds error messages — assert it straight off the tree across a multi-line input.
    tree = parse("1 +\n2 *\n3")  # '+' on line 1, '*' and its left on line 2, '3' on line 3
    assert isinstance(tree, BinOp) and tree.op == "+" and tree.line == 1
    assert tree.left.line == 1  # Number "1"
    mul = tree.right
    assert isinstance(mul, BinOp) and mul.op == "*" and mul.line == 2
    assert mul.left.line == 2  # Number "2"
    assert mul.right.line == 3  # Number "3"


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("", 1),  # empty input
        ("   ", 1),
        ("\n\n", 3),
        ("(1+2", 1),  # unbalanced parens
        ("(1+\n2", 2),
        ("1+2)", 1),  # trailing garbage
        ("1 2", 1),
        ("()", 1),  # operator/atom missing
        ("1+", 1),
        ("1+\n", 2),
        ("*3", 1),
        ("1//", 1),
        ("2**", 1),
        ("foo(1)", 1),  # unknown function name (22.2)
        ("sqrt(1, 2)", 1),  # wrong arity
        ("sqrt()", 1),  # missing argument -> the atom parser fires
        ("sqrt", 1),  # bare name, no call parens
        ("sqrt 4", 1),  # name not followed by '('
        ("sqrt(4", 1),  # unclosed call
        ("1 +\nfoo(2)", 2),  # unknown name's line is the NAME's, not the call's first arg
        ("1 +\nsqrt(2, 3)", 2),  # arity error carries the NAME's line
        ("pi(1)", 1),  # nullary given an argument (29.2) — arity 0, so 1 is too many
        ("1 +\npi(1)", 2),  # the nullary arity error carries the NAME's line too
    ],
)
def test_parse_errors_carry_line(text, line):
    with pytest.raises(ParseError) as excinfo:
        parse(text)
    assert excinfo.value.line == line


def test_empty_input_message():
    with pytest.raises(ParseError, match="empty input"):
        parse("  ")


def test_unbalanced_paren_message():
    with pytest.raises(ParseError, match=r"expected '\)'"):
        parse("(1+2")


def test_trailing_garbage_message():
    with pytest.raises(ParseError, match="after the expression"):
        parse("1 2")


def test_lex_errors_propagate_from_parse():
    with pytest.raises(LexError) as excinfo:
        parse("1 +\n$")
    assert excinfo.value.line == 2


def test_function_call_parses_to_funccall():
    assert parse("sqrt(4)") == FuncCall("sqrt", (Number("4", line=1),), line=1)


def test_function_call_argument_is_a_full_expression():
    # The argument is parsed by expression(), so it carries the whole precedence stack.
    assert parse("sqrt(1 + 2 * 3)") == FuncCall(
        "sqrt",
        (
            BinOp(
                "+",
                Number("1", line=1),
                BinOp("*", Number("2", line=1), Number("3", line=1), line=1),
                line=1,
            ),
        ),
        line=1,
    )


def test_pretty_golden_function_call():
    assert parse("sqrt(2 * 8) + 1").pretty() == (
        "BINARY_ADD\n"
        '  CALL "sqrt"\n'
        "    BINARY_MUL\n"
        '      LITERAL "2"\n'
        '      LITERAL "8"\n'
        '  LITERAL "1"'
    )


def test_function_call_is_an_atom_binds_tighter_than_power():
    # sqrt(4)**2 is (sqrt(4))**2 — the call is an atom, tighter than ** (22.2).
    assert parse("sqrt(4)**2") == BinOp(
        "**", FuncCall("sqrt", (Number("4", line=1),), line=1), Number("2", line=1), line=1
    )


def test_unary_minus_wraps_a_function_call():
    assert parse("-sqrt(4)") == UnaryOp(
        "-", FuncCall("sqrt", (Number("4", line=1),), line=1), line=1
    )


def test_unknown_function_message():
    with pytest.raises(ParseError, match="unknown function 'foo'"):
        parse("foo(1)")


def test_wrong_arity_message():
    with pytest.raises(ParseError, match=r"'sqrt' takes 1 argument\(s\), but 2 given"):
        parse("sqrt(1, 2)")


def test_variadic_call_accepts_any_count():
    # sum is variadic (>= 1 arg), so the parser accepts one, two, or many args
    # where a fixed-arity func would reject all but its exact count.
    one = Number("1", line=1)
    assert parse("sum(1)") == FuncCall("sum", (one,), line=1)
    assert parse("sum(1, 1, 1)") == FuncCall("sum", (one, one, one), line=1)


def test_variadic_min_arity_message():
    # A function with a minimum arity phrases the bound as "at least N". sum can
    # only be under-supplied via direct construction — empty sum() hits the atom
    # parser's "expected a number" first — so assert the phrasing through FuncCall.
    with pytest.raises(ValueError, match=r"'sum' takes at least 1 argument\(s\), got 0"):
        FuncCall("sum", (), line=1)


def test_nullary_call_parses_with_an_empty_arg_list():
    # pi()/e() are nullaries (29.2): NAME '(' ')' with NO arguments, the one call
    # shape where the empty parens are legal. The args tuple is empty, the line the
    # NAME's — same FuncCall node as every other call, just arity 0.
    assert parse("pi()") == FuncCall("pi", (), line=1)
    assert parse("e()") == FuncCall("e", (), line=1)


def test_nullary_call_is_an_atom_binds_tighter_than_power():
    # 2*pi() is 2*(pi()) and pi()**2 is (pi())**2 — a nullary call is an atom like
    # any other, so it binds tighter than * and ** despite the empty arg list (29.2).
    assert parse("2 * pi()") == BinOp("*", Number("2", line=1), FuncCall("pi", (), line=1), line=1)
    assert parse("pi()**2") == BinOp("**", FuncCall("pi", (), line=1), Number("2", line=1), line=1)


def test_nullary_wrong_arity_message():
    # A nullary takes 0 args, so any argument is a wrong-count parse error (29.4) —
    # the fixed-arity phrasing, "0 argument(s)", mirroring sqrt's wrong-arity message.
    with pytest.raises(ParseError, match=r"'pi' takes 0 argument\(s\), but 1 given"):
        parse("pi(1)")


def test_nullary_call_evaluates_end_to_end():
    # pi() is wired through nodes._NULLARY_FUNCS to Value.pi, dispatched the context
    # rather than operands (29.2). Irrational, so floating-point is inexact and
    # rational refuses; here just prove the float path reaches math.pi.
    import math

    result = parse("pi()").evaluate(Mode.FLOATING_POINT)
    assert result.payload == math.pi
    assert result.exact is False


def test_function_call_evaluates_end_to_end():
    # sqrt is wired through nodes._FUNCS to Value.sqrt; a perfect square is exact.
    result = parse("sqrt(16) + 1").evaluate(Mode.FIXED_POINT)
    assert result.to_string() == "5"
    assert result.exact is True
    # binary64 sqrt is unconditionally inexact.
    assert parse("sqrt(2)").evaluate(Mode.FLOATING_POINT).exact is False


def test_abs_call_evaluates_end_to_end():
    # 22.4.1 plumbing: abs parses as a call and dispatches to Value.abs_ (the call
    # name is "abs"; the method's trailing underscore is internal). Negatives reach
    # it through unary minus, so this exercises the whole 22.1-22.3 chain per mode.
    assert parse("abs(-3)").evaluate(Mode.RATIONAL).payload == Fraction(3)
    assert parse("abs(2 - 5)").evaluate(Mode.FIXED_POINT).to_string() == "3"
    assert parse("abs(-3.5)").evaluate(Mode.FLOATING_POINT).payload == 3.5
    # abs is exact in every mode, so an exact operand stays exact.
    assert parse("abs(-3)").evaluate(Mode.RATIONAL).exact is True


def test_parse_then_evaluate_end_to_end():
    assert parse("(1 + 2 * 10**3) / 4").evaluate(Mode.RATIONAL).payload == Fraction(2001, 4)
    assert parse("-2**2").evaluate(Mode.RATIONAL).payload == Fraction(-4)
    # Base-prefixed lexemes are plain integer literals in every mode (20.4).
    assert parse("0xFF - 0b1 - 0o17").evaluate(Mode.RATIONAL).payload == Fraction(239)
    assert parse("0x10 + 0.5").evaluate(Mode.FLOATING_POINT).payload == 16.5


# --- end-to-end FIXED_POINT at 18 decimals, billions-scale (ERC-20 idiom) ---
# A full-width ERC-20 amount is a bignum mantissa: 1 token == 10**18 base units,
# so a billion tokens is 10**9 * 10**18 == 10**27 units. These parse-then-evaluate
# cases prove the engine stays exact to the last unit where the result fits the
# 18-decimal scale, and rounds half-to-even (flagging inexact) where it does not.


def _fp18(text):
    return parse(text).evaluate(Mode.FIXED_POINT)


def test_fixed_point_billions_add_is_exact_to_the_unit():
    # 2.5 billion + 1 unit, plus 1.5 billion - 1 unit == exactly 4 billion.
    result = _fp18("2500000000.000000000000000001 + 1499999999.999999999999999999")
    assert result.payload == FixedPoint(4 * 10**27, 18)
    assert result.exact is True


def test_fixed_point_billions_sub_keeps_full_precision():
    # 3 billion tokens minus a single base unit (10**-18) — no float could hold this.
    result = _fp18("3000000000.000000000000000000 - 0.000000000000000001")
    assert result.payload == FixedPoint(3 * 10**27 - 1, 18)
    assert result.exact is True


def test_fixed_point_billions_mul_total_supply_is_exact():
    # Per-holder allocation * holder count == total supply, exact at 18 decimals.
    result = _fp18("1250000000.000000000000000000 * 8.000000000000000000")
    assert result.payload == FixedPoint(10**28, 18)  # 10 billion tokens
    assert result.exact is True


def test_fixed_point_billions_pow_stays_bignum_exact():
    # (2 billion)**2 == 4*10**18 tokens; the mantissa carries 18 more decimals.
    result = _fp18("2000000000.000000000000000000 ** 2")
    assert result.payload == FixedPoint(4 * 10**18 * 10**18, 18)
    assert result.exact is True


def test_fixed_point_billions_mul_rounds_half_even_and_flags_inexact():
    # The true product has digits past 18 decimals; they are truncated (ties to
    # even) and the result is flagged inexact rather than silently dropped.
    result = _fp18("1000000000.000000000000000001 * 1.000000000000000001")
    assert result.payload == FixedPoint(10**27 + 10**9 + 1, 18)
    assert result.exact is False


def test_fixed_point_billions_div_truncates_to_scale():
    # 1 billion / 3 does not terminate; it rounds to 18 decimals and is inexact.
    result = _fp18("1000000000.000000000000000000 / 3.000000000000000000")
    assert result.payload == FixedPoint(333_333_333 * 10**18 + (10**18 - 1) // 3, 18)
    assert result.exact is False


def test_fixed_point_billions_decimal_written_scale():
    # A decimal sets its scale by writing the digits (19.2.1) — '@' is hex-only:
    # a billion ETH (10**9 ETH == 10**27 wei) at 18 decimals, added to itself.
    result = _fp18("1000000000.000000000000000000 + 1000000000.000000000000000000")
    assert result.payload == FixedPoint(2 * 10**27, 18)
    assert result.exact is True


def test_fixed_point_billions_at_notation_hex_mantissa():
    # 1 ETH as the canonical hex wei literal (0xDE0B6B3A7640000 == 10**18),
    # scaled up to 5 billion ETH — exercises base-prefixed M@D end-to-end.
    result = _fp18("0xDE0B6B3A7640000@18 * 5000000000.000000000000000000")
    assert result.payload == FixedPoint(5 * 10**9 * 10**18, 18)
    assert result.exact is True


def test_fixed_point_billions_combined_expression_is_exact():
    # supply*price then dock a single base-unit fee — precedence and exactness
    # both hold across a billions-scale compound expression.
    result = _fp18("(2000000000.000000000000000000 * 3.000000000000000000) - 0.000000000000000001")
    assert result.payload == FixedPoint(6 * 10**27 - 1, 18)
    assert result.exact is True


# --- end-to-end FIXED_POINT with long Ethereum-style hex wei literals --------
# On-chain, an amount is a uint256 of wei (1 ETH == 10**18 wei), almost always
# written in hex. The M@D notation `0x<wei>@18` is exactly that: a hex wei
# mantissa tagged with the 18-decimal ERC-20 scale, so a balance reads as the
# raw on-chain word yet evaluates as the ETH value. These cases push full-width
# hex literals through parse-then-evaluate and check the result to the wei.


def test_fixed_point_hex_wei_add_two_balances():
    # 0xD8D726B7177A800000 == 4000 ETH, 0x3635C9ADC5DEA00000 == 1000 ETH.
    result = _fp18("0xD8D726B7177A800000@18 + 0x3635C9ADC5DEA00000@18")
    assert result.payload == FixedPoint(5000 * 10**18, 18)  # exactly 5000 ETH
    assert result.exact is True


def test_fixed_point_hex_wei_sub_transfer():
    # 0xD3C21BCECCEDA1000000 == 1,000,000 ETH; debit 0x6B14E9F7E4F5A5000
    # == 123.456789 ETH, leaving 999876.543211 ETH exact to the wei.
    result = _fp18("0xD3C21BCECCEDA1000000@18 - 0x6B14E9F7E4F5A5000@18")
    assert result.payload == FixedPoint(999_876_543_211 * 10**12, 18)
    assert result.exact is True


def test_fixed_point_hex_wei_gas_fee_times_integer_count():
    # gasPrice 0x4A817C800 wei (20 gwei) * 21000 gas units == 0.00042 ETH.
    # The plain integer gas count has scale 0; the result unifies to 18 decimals.
    result = _fp18("0x4A817C800@18 * 21000")
    assert result.payload == FixedPoint(21_000 * 20 * 10**9, 18)  # 420000000000000 wei
    assert result.exact is True


def test_fixed_point_hex_wei_total_supply_scaled():
    # 0x33B2E3C9FD0803CE8000000 == 10**9 tokens (a billion, 18-dec), doubled.
    result = _fp18("0x33B2E3C9FD0803CE8000000@18 * 2.000000000000000000")
    assert result.payload == FixedPoint(2 * 10**27, 18)  # 2 billion tokens
    assert result.exact is True


def test_fixed_point_hex_wei_div_rounds_half_even_inexact():
    # 0x56BC75E2D63100000 == 100 ETH split 7 ways — does not terminate, so it
    # rounds to 18 decimals (ties to even) and is flagged inexact.
    result = _fp18("0x56BC75E2D63100000@18 / 7.000000000000000000")
    assert result.payload == FixedPoint(14_285_714_285_714_285_714, 18)
    assert result.exact is False


def test_fixed_point_hex_wei_balance_minus_gas_fee():
    # Precedence holds over long hex: balance - (gasPrice * gasUsed). 5 ETH less
    # a 21000-gas fee at 20 gwei == 4.99958 ETH, exact to the wei.
    result = _fp18("0x4563918244F40000@18 - 0x4A817C800@18 * 21000")
    assert result.payload == FixedPoint(5 * 10**18 - 21_000 * 20 * 10**9, 18)
    assert result.exact is True


def test_reexports_from_expr_package():
    import mcp_abacus.expr as expr
    import mcp_abacus.expr.lexer as lexer
    import mcp_abacus.expr.parser as parser_module

    assert expr.parse is parser_module.parse
    assert expr.ParseError is parser_module.ParseError
    assert expr.tokenize is lexer.tokenize
    assert expr.Token is lexer.Token
    assert expr.LexError is lexer.LexError
