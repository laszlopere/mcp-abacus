# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit tests for the universal Value class (TODO 19) — its full public
contract for the one mode built so far, FLOATING_POINT (19.1.1).

Organised by the concerns recorded in value.py's docstring: the Mode enum,
construction/storage, from_lexeme, the binary and unary operators, exactness
propagation, "no mode mixing / no bare numbers", and the mode-agnostic domain
errors. As further Mode members land (19.1.2+), each section gains that mode's
cases beside FLOATING_POINT's.
"""

import dataclasses
import math
import struct
from enum import Enum
from fractions import Fraction

import pytest

from mcp_abacus.expr.value import (
    INEXACT_HANDLING_HELP,
    FixedPoint,
    InexactHandling,
    Mode,
    NotRepresentableError,
    Value,
    _fp_ln,
    _ln2_scaled,
    _ln10_scaled,
    _pi_scaled,
    resolve_inexact_handling,
)

# Named methods, no operator overloading (decision 2026-06-12) — so the ops are
# addressed by name. These drive the parametrized op-agnostic tests below.
BINARY_OPS = ("add", "sub", "mul", "div", "floordiv", "mod", "pow", "bitand", "bitor", "bitxor")
UNARY_OPS = ("neg", "pos", "bitnot")


def _v(x: float, exact: bool = False) -> Value:
    return Value(Mode.FLOATING_POINT, x, exact=exact)


def _f64(lexeme: str) -> Value:
    return Value.from_lexeme(lexeme, Mode.FLOATING_POINT)


def _rat(lexeme: str) -> Value:
    return Value.from_lexeme(lexeme, Mode.RATIONAL)


def _fp(lexeme: str) -> Value:
    return Value.from_lexeme(lexeme, Mode.FIXED_POINT)


# --- the Mode enum (19.1) --------------------------------------------------


def test_mode_is_a_plain_enum():
    # A plain enum, one member per concrete type — no ModeSpec (decision settled).
    assert issubclass(Mode, Enum)


def test_built_modes_exist():
    assert list(Mode) == [Mode.FLOATING_POINT, Mode.FIXED_POINT, Mode.RATIONAL]
    assert Mode.FLOATING_POINT.value == "floating-point"
    assert Mode.FIXED_POINT.value == "fixed-point"
    assert Mode.RATIONAL.value == "rational"


# --- construction & storage (19.1.1) ---------------------------------------


def test_value_holds_a_double():
    v = Value(Mode.FLOATING_POINT, 3.5, exact=False)
    assert v.mode is Mode.FLOATING_POINT
    assert v.payload == 3.5
    assert type(v.payload) is float
    assert v.exact is False


def test_exact_flag_is_stored_as_given():
    assert Value(Mode.FLOATING_POINT, 1.0, exact=True).exact is True
    assert Value(Mode.FLOATING_POINT, 1.0, exact=False).exact is False


def test_value_is_frozen():
    v = _v(1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.payload = 2.0  # type: ignore[misc]


def test_value_uses_slots_so_has_no_instance_dict():
    # slots=True — no per-instance __dict__; attributes are fixed.
    assert not hasattr(_v(1.0), "__dict__")


@pytest.mark.parametrize("bad_payload", [3, "3.5", True], ids=["int", "str", "bool"])
def test_floating_point_payload_must_be_exactly_float(bad_payload):
    # Storage validation is `type(payload) is float`: int, str, and even bool
    # (an int subclass) are all rejected — only a true float is FLOATING_POINT storage.
    with pytest.raises(ValueError, match="FLOATING_POINT payload must be a float"):
        Value(Mode.FLOATING_POINT, bad_payload, exact=False)  # type: ignore[arg-type]


def test_unknown_mode_rejected_at_construction():
    with pytest.raises(ValueError, match="unsupported mode"):
        Value("floating-point", 3.5, exact=False)  # type: ignore[arg-type]


# --- from_lexeme (19.2) ----------------------------------------------------


@pytest.mark.parametrize(
    "lexeme, expected",
    [
        ("42", 42.0),  # integer
        ("0.1", 0.1),  # decimal
        ("3.", 3.0),  # trailing-dot decimal
        (".5", 0.5),  # leading-dot decimal
        ("1e-3", 1e-3),  # scientific
        ("2.5E+6", 2_500_000.0),  # scientific, uppercase E, explicit +
    ],
)
def test_from_lexeme_decimal_and_scientific_forms(lexeme, expected):
    # Every non-prefixed 20.1.3 literal form, interpreted in floating_point's repr.
    assert _f64(lexeme).payload == expected


@pytest.mark.parametrize(
    "lexeme, expected",
    [
        ("0xFF", 255.0),
        ("0b1010", 10.0),
        ("0o17", 15.0),
        ("0X1F", 31.0),  # uppercase prefix accepted
        ("0B101", 5.0),
        ("0O20", 16.0),
    ],
)
def test_from_lexeme_base_prefixed_integers(lexeme, expected):
    # 20.4: a base-prefixed lexeme is an INTEGER literal, held here as a double.
    # float() rejects "0xFF" outright, so from_lexeme must parse the base first.
    assert _f64(lexeme).payload == expected


def test_from_lexeme_result_is_a_floating_point_float():
    for lexeme in ("42", "0.1", "0xFF"):
        v = _f64(lexeme)
        assert v.mode is Mode.FLOATING_POINT
        assert type(v.payload) is float


@pytest.mark.parametrize("lexeme", ["", "hello", "0xZZ", "1.2.3"])
def test_from_lexeme_rejects_garbage(lexeme):
    # A lexeme that is neither a base-prefixed integer nor a valid float is an
    # error (propagated from int()/float()); not a silent zero or NaN.
    with pytest.raises(ValueError):
        _f64(lexeme)


def test_from_lexeme_is_unconditionally_inexact():
    # floating_point carries exact=False for now (provisional — 1.0+2.0 IS exact).
    assert _f64("0.1").exact is False
    assert _f64("42").exact is False


# --- binary operators (19.3.1-19.3.7) --------------------------------------


def test_add():
    assert _v(6.0).add(_v(4.0)).payload == 10.0


def test_sub():
    assert _v(6.0).sub(_v(4.0)).payload == 2.0


def test_mul():
    assert _v(6.0).mul(_v(4.0)).payload == 24.0


def test_div():
    assert _v(6.0).div(_v(4.0)).payload == 1.5


def test_floordiv():
    assert _v(7.0).floordiv(_v(2.0)).payload == 3.0


def test_mod():
    assert _v(7.0).mod(_v(2.0)).payload == 1.0


def test_pow_is_power_not_xor():
    # ** routes here as POWER: 2**10 == 1024, NOT Python's xor (2 ^ 10 == 8).
    assert _v(2.0).pow(_v(10.0)).payload == 1024.0


def test_pow_negative_exponent_and_zero_zero():
    assert _v(2.0).pow(_v(-2.0)).payload == 0.25
    assert _v(0.0).pow(_v(0.0)).payload == 1.0  # follows Python


@pytest.mark.parametrize(
    "op, dividend, divisor, expected",
    [
        ("mod", -7.0, 3.0, 2.0),  # Python's sign-of-divisor remainder
        ("mod", 7.0, -3.0, -2.0),
        ("floordiv", -7.0, 3.0, -3.0),  # floors toward -inf, not truncation
    ],
)
def test_div_family_follows_python_sign_rules(op, dividend, divisor, expected):
    # Faithfulness: //, % keep Python/IEEE semantics for negative operands.
    assert getattr(_v(dividend), op)(_v(divisor)).payload == expected


@pytest.mark.parametrize("op", BINARY_OPS)
def test_binary_op_returns_a_new_floating_point_value(op):
    a, b = _v(7.0), _v(2.0)
    r = getattr(a, op)(b)
    assert r.mode is Mode.FLOATING_POINT
    assert type(r.payload) is float
    assert r is not a and r is not b  # immutable: a fresh Value


# --- unary operators (19.3.8-19.3.9) ---------------------------------------


def test_neg():
    assert _v(3.5).neg().payload == -3.5
    assert _v(-3.5).neg().payload == 3.5


def test_pos_is_identity_on_value():
    assert _v(-3.5).pos().payload == -3.5
    assert _v(3.5).pos().payload == 3.5


@pytest.mark.parametrize("op", UNARY_OPS)
def test_unary_op_returns_new_value_preserving_mode(op):
    v = _v(3.5)
    r = getattr(v, op)()
    assert r.mode is Mode.FLOATING_POINT
    assert type(r.payload) is float
    assert r is not v


# --- exactness propagation -------------------------------------------------


@pytest.mark.parametrize("op", BINARY_OPS)
def test_binary_exactness_is_and_of_both_operands(op):
    # Result is exact only if EVERY operand was. Operands chosen so every op
    # (incl. div/floordiv/mod/pow) computes without a domain error.
    assert getattr(_v(6.0, exact=True), op)(_v(2.0, exact=True)).exact is True
    assert getattr(_v(6.0, exact=True), op)(_v(2.0, exact=False)).exact is False
    assert getattr(_v(6.0, exact=False), op)(_v(2.0, exact=True)).exact is False
    assert getattr(_v(6.0, exact=False), op)(_v(2.0, exact=False)).exact is False


@pytest.mark.parametrize("op", UNARY_OPS)
def test_unary_ops_preserve_exactness(op):
    assert getattr(_v(1.0, exact=True), op)().exact is True
    assert getattr(_v(1.0, exact=False), op)().exact is False


# --- no mode mixing / no bare numbers --------------------------------------


@pytest.mark.parametrize("op", BINARY_OPS)
def test_every_binary_op_rejects_a_bare_python_number(op):
    # "operators take Values, nothing else" — no promotion from plain numbers.
    with pytest.raises(TypeError, match="requires a Value"):
        getattr(_v(1.0), op)(2.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("op", BINARY_OPS)
def test_every_binary_op_rejects_a_different_mode(op):
    # "No mode mixing" — both operands must share a mode; no silent promotion
    # between floating_point and rational. (Now reachable: two modes exist.)
    with pytest.raises(TypeError, match="mode mismatch"):
        getattr(_rat("1"), op)(_v(2.0))


# --- domain errors (mode-agnostic, no line info) ---------------------------


@pytest.mark.parametrize("op", ["div", "floordiv", "mod"])
def test_division_by_zero_raises_arithmetic_error_without_line(op):
    # Provisional rule: follows Python (raises ZeroDivisionError), not IEEE
    # ±inf. An ArithmeticError so BinOp.evaluate (18.4) catches it; raised here
    # carrying NO position info — that is attached later by the node.
    with pytest.raises(ArithmeticError) as excinfo:
        getattr(_v(1.0), op)(_v(0.0))
    assert not hasattr(excinfo.value, "line")


def test_not_representable_is_an_arithmetic_error():
    # So the same catch-and-attach-line path in BinOp.evaluate handles it.
    assert issubclass(NotRepresentableError, ArithmeticError)


def test_complex_power_is_not_representable():
    # A negative base to a fractional exponent goes complex in Python; floating_point
    # cannot hold it, so it is a domain error rather than a silent complex.
    with pytest.raises(NotRepresentableError, match="complex result"):
        _v(-2.0).pow(_v(0.5))


# --- formatting (19.4) -----------------------------------------------------


def test_to_string_floating_point():
    assert _v(3.5).to_string() == "3.5"
    assert _v(42.0).to_string() == "42.0"


def test_to_string_round_trips_through_from_lexeme():
    # str(float) is the shortest round-tripping decimal (10.3), so re-parsing
    # to_string() must reproduce the exact same double.
    for lexeme in ("2.5E+6", "0.1", "42"):
        v = _f64(lexeme)
        assert _f64(v.to_string()).payload == v.payload


# --- RATIONAL mode (19.1.7) ------------------------------------------------
# The second built mode: a Fraction payload, exact by construction. Its tests
# mirror FLOATING_POINT's above, plus the flagship exactness divergence.


@pytest.mark.parametrize("bad_payload", [3, 3.5, "1/2"], ids=["int", "float", "str"])
def test_rational_payload_must_be_a_fraction(bad_payload):
    with pytest.raises(ValueError, match="RATIONAL payload must be a Fraction"):
        Value(Mode.RATIONAL, bad_payload, exact=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "lexeme, expected",
    [
        ("42", Fraction(42)),
        ("0.1", Fraction(1, 10)),  # held EXACTLY, unlike the floating_point 0.1
        ("3.", Fraction(3)),
        (".5", Fraction(1, 2)),
        ("1e-3", Fraction(1, 1000)),
        ("2.5E+6", Fraction(2_500_000)),
        ("0xFF", Fraction(255)),  # base-prefixed integer, valid in every mode
        ("0b1010", Fraction(10)),
        ("0o17", Fraction(15)),
    ],
)
def test_from_lexeme_rational_is_an_exact_fraction(lexeme, expected):
    v = _rat(lexeme)
    assert v.payload == expected
    assert type(v.payload) is Fraction
    assert v.mode is Mode.RATIONAL
    assert v.exact is True  # every literal is exact in rational mode


def test_rational_arithmetic():
    assert _rat("1").add(_rat("2")).payload == Fraction(3)
    assert _rat("1").sub(_rat("2")).payload == Fraction(-1)
    assert _rat("2").mul(_rat("3")).payload == Fraction(6)
    assert _rat("1").div(_rat("3")).payload == Fraction(1, 3)  # stays exact, no rounding
    assert _rat("7").mod(_rat("2")).payload == Fraction(1)
    assert _rat("2").pow(_rat("10")).payload == Fraction(1024)
    assert _rat("2").pow(_rat("-2")).payload == Fraction(1, 4)  # negative integer exponent


def test_rational_floordiv_returns_a_fraction():
    # Fraction // Fraction is an int; the result is re-wrapped as a Fraction.
    r = _rat("7").floordiv(_rat("2"))
    assert r.payload == Fraction(3)
    assert type(r.payload) is Fraction


def test_rational_power_requires_an_integer_exponent():
    # A non-integer exponent would be irrational — not representable as a Fraction.
    with pytest.raises(NotRepresentableError, match="integer exponent"):
        _rat("2").pow(_rat("0.5"))


def test_rational_arithmetic_stays_exact():
    assert _rat("1").add(_rat("2")).exact is True
    assert _rat("1").div(_rat("3")).exact is True


def test_flagship_divergence_rational_is_exact_where_floating_point_is_not():
    # THE core concept: 0.1 + 0.2. Rational holds it exactly; floating_point does not.
    rational = _rat("0.1").add(_rat("0.2"))
    assert rational.payload == Fraction(3, 10)
    assert rational.exact is True

    floating_point = _f64("0.1").add(_f64("0.2"))
    assert floating_point.payload == 0.30000000000000004
    assert floating_point.payload != 0.3
    assert floating_point.exact is False


def test_to_string_rational():
    assert _rat("1500").to_string() == "1500"  # integral Fraction -> "n"
    assert _rat("0.5").to_string() == "1/2"  # non-integral -> "n/d"


# --- FIXED_POINT mode (19.1.2) ---------------------------------------------
# The flagship: a scaled integer (mantissa, decimals). Exact by construction;
# binary ops carry the result at scale max(d1, d2) — the precision that covers
# both operands — rounding half-to-even (and flagging inexact) when it does not
# fit. mantissa is a Python int (bignum), so full-width ERC-20 amounts are safe.


def test_fixed_point_payload_must_be_a_fixedpoint():
    for bad in (3, 3.5, "1.5", Fraction(3, 2)):
        with pytest.raises(ValueError, match="FIXED_POINT payload must be a FixedPoint"):
            Value(Mode.FIXED_POINT, bad, exact=True)  # type: ignore[arg-type]


def test_fixedpoint_rejects_bad_components():
    with pytest.raises(ValueError, match="mantissa must be an int"):
        FixedPoint(1.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mantissa must be an int"):
        FixedPoint(True, 0)  # bool is an int subclass — still rejected
    with pytest.raises(ValueError, match="decimals must be a non-negative int"):
        FixedPoint(15, -1)


@pytest.mark.parametrize(
    "lexeme, mantissa, decimals",
    [
        ("42", 42, 0),  # integer -> scale 0
        ("1.5", 15, 1),  # decimal
        ("1.50", 150, 2),  # trailing zero is SIGNIFICANT (scale 2, not 1)
        ("0.5", 5, 1),
        (".5", 5, 1),  # leading-dot decimal
        ("3.", 3, 0),  # trailing-dot decimal is a whole number
        ("1e-3", 1, 3),  # scientific with negative exponent -> scale 3
        ("2.5E+6", 2_500_000, 0),  # scientific scaling up -> whole number
        ("0xFF", 255, 0),  # base-prefixed integer, valid in every mode (20.4)
        ("0b1010", 10, 0),
    ],
)
def test_from_lexeme_fixed_point_scaled_integer(lexeme, mantissa, decimals):
    v = _fp(lexeme)
    assert v.payload == FixedPoint(mantissa, decimals)
    assert v.mode is Mode.FIXED_POINT
    assert v.exact is True  # every literal is exact in fixed-point


@pytest.mark.parametrize(
    "lexeme, mantissa, decimals",
    [
        ("0x3039@2", 12345, 2),  # 123.45 (0x3039 == 12345)
        ("0x59682F00@9", 1_500_000_000, 9),  # 1.5 at 9 decimals
        ("0xDE0B6B3A7640000@18", 10**18, 18),  # 1 ETH in wei
        ("0x64@0", 100, 0),  # '@0' is the bare integer (0x64 == 100)
    ],
)
def test_from_lexeme_fixed_point_at_notation(lexeme, mantissa, decimals):
    # M@D == M * 10**-D (20.5): the mantissa (any base) keeps its declared scale.
    v = _fp(lexeme)
    assert v.payload == FixedPoint(mantissa, decimals)
    assert v.exact is True


def test_at_notation_is_the_same_value_in_every_mode():
    # The side-by-side product: '0x59682F00@9' is 1.5 in EVERY mode, not just
    # fixed-point — like base-prefixed integers, M@D evaluates in all modes.
    assert _fp("0x59682F00@9").to_string() == "1.500000000"
    assert _rat("0x59682F00@9").payload == Fraction(3, 2)
    assert _f64("0x59682F00@9").payload == 1.5


@pytest.mark.parametrize(
    "mantissa, decimals, text",
    [
        (42, 0, "42"),  # whole number, no point
        (15, 1, "1.5"),
        (150, 2, "1.50"),  # trailing zeros preserved (declared scale shown)
        (5, 1, "0.5"),  # sub-1 gets a leading zero
        (3, 2, "0.03"),  # zero-padded fraction
        (0, 0, "0"),
        (0, 2, "0.00"),
        (-150, 2, "-1.50"),  # sign sits before the digits
        (-5, 1, "-0.5"),
        (12345, 2, "123.45"),
    ],
)
def test_to_string_fixed_point(mantissa, decimals, text):
    assert Value(Mode.FIXED_POINT, FixedPoint(mantissa, decimals), exact=True).to_string() == text


def test_to_string_fixed_point_round_trips_through_from_lexeme():
    # Re-parsing to_string() must reproduce the same (mantissa, scale) exactly,
    # trailing zeros and all.
    for lexeme in ("1.50", "123.45", "0.001", "42", "1.000000000"):
        v = _fp(lexeme)
        assert _fp(v.to_string()).payload == v.payload


# --- precision() (25.1) ----------------------------------------------------


@pytest.mark.parametrize(
    "mantissa, decimals",
    [(150, 2), (3, 1), (42, 0), (1, 9)],
)
def test_precision_fixed_point_is_its_decimal_scale(mantissa, decimals):
    # Fixed-point's precision is its declared scale — independent of the value
    # or its exactness (a whole-number fixed-point still has a known scale of 0).
    v = Value(Mode.FIXED_POINT, FixedPoint(mantissa, decimals), exact=True)
    assert v.precision() == decimals


def test_precision_is_none_for_modes_without_a_decimal_scale():
    # Floating-point and rational carry no per-value decimal scale: "when known".
    assert Value(Mode.FLOATING_POINT, 1.5, exact=False).precision() is None
    assert Value(Mode.RATIONAL, Fraction(1, 3), exact=True).precision() is None


# --- min_fixed_point_precision floor on from_lexeme (25.2.1) ----------------


@pytest.mark.parametrize(
    "lexeme, floor, mantissa, decimals",
    [
        ("42", 4, 420000, 4),  # whole number padded from scale 0 to 4
        ("0", 3, 0, 3),  # zero too — same value, wider declared scale
        ("1.5", 4, 15000, 4),  # decimal raised from scale 1 to 4
        ("0x64", 2, 10000, 2),  # base-prefixed integer (0x64 == 100) -> 100.00
        ("0x59682F00@9", 18, 1_500_000_000_000_000_000, 18),  # M@D raised 9 -> 18
        ("12", 0, 12, 0),  # a zero floor is a no-op
    ],
)
def test_from_lexeme_fixed_point_floor_raises_scale(lexeme, floor, mantissa, decimals):
    # The floor pads zeros to reach min_fixed_point_precision decimals; the value
    # is unchanged (so it stays exact) but its declared scale rises to the floor.
    v = Value.from_lexeme(lexeme, Mode.FIXED_POINT, floor)
    assert v.payload == FixedPoint(mantissa, decimals)
    assert v.exact is True
    assert v.precision() == decimals


@pytest.mark.parametrize("lexeme, natural", [("1.23456789", 8), ("1.50", 2), ("1e-6", 6)])
def test_from_lexeme_fixed_point_floor_is_a_minimum_never_truncates(lexeme, natural):
    # A literal already wider than the floor keeps its own (larger) scale: the
    # floor is max(natural, floor), never a ceiling that would drop precision.
    v = Value.from_lexeme(lexeme, Mode.FIXED_POINT, natural - 1)
    assert v.precision() == natural


def test_from_lexeme_floor_ignored_outside_fixed_point():
    # Floating-point and rational have no decimal scale, so the floor is inert —
    # the same value as with no floor, never an error.
    assert Value.from_lexeme("1.5", Mode.FLOATING_POINT, 9).payload == 1.5
    assert Value.from_lexeme("1/2", Mode.RATIONAL, 9).payload == Fraction(1, 2)


def test_from_lexeme_floor_defaults_to_zero_no_behaviour_change():
    # The default (no floor) reproduces the pre-25.2.1 scale exactly.
    assert Value.from_lexeme("42", Mode.FIXED_POINT).payload == FixedPoint(42, 0)


# --- fixed-point operators: differing scales unify to max(d1, d2) ----------


@pytest.mark.parametrize(
    "op, a, b, mantissa, decimals",
    [
        # add/sub align to the larger scale and stay exact.
        ("add", "1.5", "0.25", 175, 2),  # scales 1 and 2 -> 2
        ("add", "1.50", "1.5", 300, 2),  # 1.50 + 1.50 == 3.00, scale kept at 2
        ("sub", "1.5", "0.25", 125, 2),
        ("add", "0.1", "0.2", 3, 1),  # the flagship: exact, unlike floating_point
        # mul: exact when the product fits the covering scale.
        ("mul", "1.50", "1.50", 225, 2),  # 2.25 fits 2 decimals
        # whole-number ops.
        ("floordiv", "7.5", "2.0", 30, 1),  # floor(3.75) == 3 -> 3.0 at scale 1
        ("mod", "7.5", "2.0", 15, 1),  # 7.5 - 3*2.0 == 1.5
        ("pow", "2.00", "3", 800, 2),  # 8.00 at base's scale
    ],
)
def test_fixed_point_exact_ops(op, a, b, mantissa, decimals):
    r = getattr(_fp(a), op)(_fp(b))
    assert r.payload == FixedPoint(mantissa, decimals)
    assert r.exact is True


@pytest.mark.parametrize(
    "op, a, b, text",
    [
        ("mul", "1.5", "1.5", "2.2"),  # 2.25 -> 1 decimal, half-even -> 2.2
        ("div", "1.0", "4.0", "0.2"),  # 0.25 -> 1 decimal, half-even -> 0.2
        ("div", "3.0", "4.0", "0.8"),  # 0.75 -> 1 decimal, half-even -> 0.8
        ("div", "1.00", "3.00", "0.33"),  # 0.333... truncated to 2 decimals
        ("pow", "1.5", "2", "2.2"),  # 2.25 -> 1 decimal
    ],
)
def test_fixed_point_inexact_ops_round_half_even(op, a, b, text):
    # When the true result does not fit the covering scale it is rounded
    # half-to-even and flagged inexact — the calculator tells the truth.
    r = getattr(_fp(a), op)(_fp(b))
    assert r.to_string() == text
    assert r.exact is False


@pytest.mark.parametrize(
    "dividend, divisor, expected_text",
    [
        ("7.5", "2.0", "1.5"),  # positive
        ("-7.5", "2.0", "0.5"),  # Python sign-of-divisor remainder
        ("7.5", "-2.0", "-0.5"),
    ],
)
def test_fixed_point_mod_follows_python_sign_rules(dividend, divisor, expected_text):
    # '-7.5' / '7.5 -2.0' build the negative via the unary minus, as the lexer
    # keeps literals unsigned. Remainder takes the divisor's sign, like floating_point.
    d = _fp(dividend[1:]).neg() if dividend.startswith("-") else _fp(dividend)
    v = _fp(divisor[1:]).neg() if divisor.startswith("-") else _fp(divisor)
    assert d.mod(v).to_string() == expected_text


def test_fixed_point_large_erc20_amount_is_exact():
    # 1.5 ETH + 0.5 ETH at 18 decimals, full wei precision, no float error.
    total = _fp("0x14D1120D7B160000@18").add(_fp("0x6F05B59D3B20000@18"))
    assert total.payload == FixedPoint(2 * 10**18, 18)  # exactly 2 ETH
    assert total.to_string() == "2.000000000000000000"
    assert total.exact is True


# --- fixed-point unary, exactness, domain errors ---------------------------


def test_fixed_point_neg_and_pos_preserve_scale():
    assert _fp("1.50").neg().payload == FixedPoint(-150, 2)
    assert _fp("1.50").neg().neg().payload == FixedPoint(150, 2)
    assert _fp("1.50").pos().payload == FixedPoint(150, 2)


@pytest.mark.parametrize("op", ["div", "floordiv", "mod"])
def test_fixed_point_division_by_zero_raises_without_line(op):
    with pytest.raises(ArithmeticError) as excinfo:
        getattr(_fp("1.0"), op)(_fp("0.0"))
    assert not hasattr(excinfo.value, "line")


def test_fixed_point_zero_to_negative_power_raises():
    with pytest.raises(ArithmeticError):
        _fp("0").pow(_fp("1").neg())


@pytest.mark.parametrize(
    "base, exponent, text",
    [
        # PATH A — a perfect q-th root lands exactly on the grid (28.20.1).
        # The result scale is max(base, exponent) — the exponent's 1 decimal here.
        ("4", "0.5", "2.0"),  # sqrt(4) at the covering scale 1
        ("0.25", "0.5", "0.50"),  # sqrt(1/4), covering scale 2 (the base's)
        ("4", "1.5", "8.0"),  # 4**(3/2) == sqrt(4**3) == 8
        ("32", "0.2", "2.0"),  # fifth root of 32 (q == 5, a product of 2s and 5s)
    ],
)
def test_fixed_point_fractional_power_exact_root(base, exponent, text):
    # A perfect root is EXACT-or-refuse like sqrt: it stays on the grid.
    r = _fp(base).pow(_fp(exponent))
    assert r.to_string() == text
    assert r.exact is True


def test_fixed_point_fractional_power_exact_negative_exponent():
    # 4**(-1/2) == 1/2; the exponent's sign is a unary op on the literal.
    r = _fp("4").pow(_fp("0.5").neg())
    assert r.to_string() == "0.5"
    assert r.exact is True


def test_fixed_point_fractional_power_exact_odd_root_of_negative():
    # Path A reaches odd roots of negative bases: (-32)**(1/5) == -2.
    r = _fp("32").neg().pow(_fp("0.2"))
    assert r.to_string() == "-2.0"
    assert r.exact is True


def test_fixed_point_fractional_power_even_root_of_negative_refuses():
    # An even root of a negative base is complex — no real value.
    with pytest.raises(NotRepresentableError, match="even root of a negative"):
        _fp("4").neg().pow(_fp("0.5"))


def test_fixed_point_fractional_power_irrational_uses_series():
    # PATH B — an irrational root computes via exp(y*ln x), inexact (28.20.1).
    r = _fp("2.000000").pow(_fp("0.5"))
    assert r.to_string() == "1.414214"  # sqrt(2), rounded to the covering scale
    assert r.exact is False


def test_fixed_point_fractional_power_negative_base_irrational_refuses():
    # An ODD irrational root of a negative base (the fifth root of -2) is real but
    # cannot go through ln(x) — refuse rather than fabricate (the x < 0 rung).
    with pytest.raises(NotRepresentableError, match="negative base is irrational"):
        _fp("2").neg().pow(_fp("0.2"))


def test_fixed_point_negative_integer_exponent():
    # x**-n is fine when it lands on the scale: 2**-1 == 0.5 at scale 1.
    assert _fp("2.0").pow(_fp("1").neg()).to_string() == "0.5"


@pytest.mark.parametrize("op", BINARY_OPS)
def test_fixed_point_binary_op_returns_a_new_fixed_point_value(op):
    a, b = _fp("7.0"), _fp("2.0")
    r = getattr(a, op)(b)
    assert r.mode is Mode.FIXED_POINT
    assert type(r.payload) is FixedPoint
    assert r is not a and r is not b


def test_flagship_divergence_fixed_point_is_exact_where_floating_point_is_not():
    # The same 0.1 + 0.2 that floating_point botches: fixed-point holds it exactly.
    fixed = _fp("0.1").add(_fp("0.2"))
    assert fixed.to_string() == "0.3"
    assert fixed.exact is True

    floating_point = _f64("0.1").add(_f64("0.2"))
    assert floating_point.payload == 0.30000000000000004
    assert floating_point.exact is False


# --- bitwise operators: & | ^ ~ on each mode's own bits (24.3.2) -----------
# Offered in EVERY mode (the decision: not integer-only — let the caller judge).
# Each mode exposes its OWN stored bits: float the 64-bit IEEE-754 pattern,
# fixed-point the scale-aligned mantissa, rational the numerator/denominator pair.
# Sign/width then follow Python's native int ops (the C/Python tool convention).


def _f64_pattern(x: float) -> bytes:
    """The raw 8-byte IEEE-754 pattern — compared instead of the float so NaN
    results (which never equal themselves) still assert reproducibly."""
    return struct.pack(">d", x)


@pytest.mark.parametrize("op, sym", [("bitand", "&"), ("bitor", "|"), ("bitxor", "^")])
def test_floating_point_bitwise_acts_on_the_64_bit_ieee_pattern(op, sym):
    a, b = 6.5, -2.25
    abits = int.from_bytes(_f64_pattern(a), "big")
    bbits = int.from_bytes(_f64_pattern(b), "big")
    combined = {"&": abits & bbits, "|": abits | bbits, "^": abits ^ bbits}[sym]
    want = struct.unpack(">d", combined.to_bytes(8, "big"))[0]
    got = getattr(_v(a), op)(_v(b)).payload
    assert _f64_pattern(got) == _f64_pattern(want)


def test_floating_point_xor_self_is_positive_zero():
    # x ^ x clears every bit -> the all-zero pattern, i.e. +0.0.
    assert _v(-3.5).bitxor(_v(-3.5)).payload == 0.0
    assert _f64_pattern(_v(-3.5).bitxor(_v(-3.5)).payload) == _f64_pattern(0.0)


def test_floating_point_bitnot_flips_all_64_bits_and_is_an_involution():
    flipped = ~int.from_bytes(_f64_pattern(1.0), "big") & ((1 << 64) - 1)
    want = struct.unpack(">d", flipped.to_bytes(8, "big"))[0]
    assert _f64_pattern(_v(1.0).bitnot().payload) == _f64_pattern(want)
    assert _v(1.0).bitnot().bitnot().payload == 1.0  # ~~x == x


@pytest.mark.parametrize(
    "op, a, b, mantissa, decimals",
    [
        # differing scales align to max() first: 1.5 & 3 == 15 & 30, not 15 & 3.
        ("bitand", "1.5", "3", 14, 1),  # 15 & 30 = 14 -> 1.4
        ("bitor", "1.5", "3", 31, 1),  # 15 | 30 = 31 -> 3.1
        ("bitxor", "1.5", "3", 17, 1),  # 15 ^ 30 = 17 -> 1.7
        # same scale: plain mantissa bit op.
        ("bitand", "12", "10", 8, 0),  # 1100 & 1010 = 1000
        # narrower operand zero-extended, wider wins (Python's non-negative default).
        ("bitor", "0x0F", "0xF0", 255, 0),  # 0x0F | 0xF0 = 0xFF
    ],
)
def test_fixed_point_bitwise_on_scale_aligned_mantissa(op, a, b, mantissa, decimals):
    r = getattr(_fp(a), op)(_fp(b))
    assert r.payload == FixedPoint(mantissa, decimals)
    assert r.exact is True  # exact bit manipulation, no rounding


@pytest.mark.parametrize(
    "lexeme, mantissa, decimals",
    [("5", -6, 0), ("1.5", -16, 1)],  # ~x == -x-1 on the mantissa, scale kept
)
def test_fixed_point_bitnot_inverts_the_mantissa_keeping_scale(lexeme, mantissa, decimals):
    r = _fp(lexeme).bitnot()
    assert r.payload == FixedPoint(mantissa, decimals)


@pytest.mark.parametrize(
    "op, a, b, expected",
    [
        ("bitxor", "1.5", "1.25", Fraction(1)),  # (3^5)/(2^4) = 6/6 = 1
        ("bitand", "12", "10", Fraction(8)),  # integers: (12&10)/(1&1) = 8/1
        ("bitor", "12", "10", Fraction(14)),  # (12|10)/(1|1) = 14/1
    ],
)
def test_rational_bitwise_is_componentwise_on_numerator_and_denominator(op, a, b, expected):
    assert getattr(_rat(a), op)(_rat(b)).payload == expected


def test_rational_bitwise_raises_when_denominators_combine_to_zero():
    # (3/2) & (5/4): den 2 & 4 == 0, which Fraction rejects — a domain error with
    # no line (BinOp.evaluate attaches it), the price of faithful componentwise bits.
    with pytest.raises(ArithmeticError) as excinfo:
        _rat("1.5").bitand(_rat("1.25"))
    assert not hasattr(excinfo.value, "line")


def test_rational_bitnot_inverts_both_components():
    # ~ is componentwise too: ~(3/2) = (~3)/(~2) = -4/-3 = 4/3; ~den is never 0.
    assert _rat("1.5").bitnot().payload == Fraction(4, 3)


# --- sqrt(): the first irrational function (19.5.2) ------------------------
# Per mode: floating-point computes and rounds (always inexact); fixed-point
# isqrt-on-the-scaled-mantissa and rounds to the operand's scale (exact only on a
# perfect square at that scale); rational is exact for a perfect square, else
# refuses (irrational, no scale to round to). Negative refuses in every mode.


def test_sqrt_floating_point_computes_and_is_inexact():
    assert _f64("2").sqrt().payload == math.sqrt(2)
    r = _f64("2").sqrt()
    assert r.mode is Mode.FLOATING_POINT
    assert r.exact is False


def test_sqrt_floating_point_is_inexact_even_for_a_perfect_square():
    # binary64 sqrt is unconditionally inexact (19.5.2) — 4 -> 2.0 still exact=False.
    r = _f64("4").sqrt()
    assert r.payload == 2.0
    assert r.exact is False


@pytest.mark.parametrize(
    "lexeme, text",
    [
        ("2.00", "1.41"),  # 1.4142... at scale 2 -> 1.41
        ("2.000000", "1.414214"),  # more scale -> more digits, rounded to nearest
        ("0.02", "0.14"),  # 0.1414... at scale 2
    ],
)
def test_sqrt_fixed_point_rounds_to_operand_scale_and_is_inexact(lexeme, text):
    r = _fp(lexeme).sqrt()
    assert r.to_string() == text
    assert r.exact is False
    assert r.precision() == _fp(lexeme).precision()  # scale preserved


@pytest.mark.parametrize(
    "lexeme, text",
    [
        ("4", "2"),  # perfect square at scale 0
        ("2.25", "1.50"),  # 1.5 lands exactly at scale 2
        ("0.0625", "0.2500"),  # 0.25 exact at scale 4
        ("0", "0"),
    ],
)
def test_sqrt_fixed_point_is_exact_on_a_perfect_square_at_scale(lexeme, text):
    r = _fp(lexeme).sqrt()
    assert r.to_string() == text
    assert r.exact is True


def test_sqrt_fixed_point_rounds_to_nearest_not_floor():
    # sqrt(3) == 1.732... at scale 0: isqrt's floor is 1, but nearest is 2, so the
    # result rounds UP — it is not a bare integer-sqrt floor.
    r = _fp("3").sqrt()
    assert r.to_string() == "2"
    assert r.exact is False


def test_sqrt_negative_refuses_in_every_mode():
    for v in (_f64("2").neg(), _fp("2").neg(), _rat("2").neg()):
        with pytest.raises(NotRepresentableError, match="negative"):
            v.sqrt()


@pytest.mark.parametrize(
    "lexeme, expected",
    [
        ("4", Fraction(2)),  # perfect square integer
        ("2.25", Fraction(3, 2)),  # 9/4 -> 3/2, both parts square
        ("0.04", Fraction(1, 5)),  # 1/25 -> 1/5
    ],
)
def test_sqrt_rational_is_exact_for_a_perfect_square(lexeme, expected):
    r = _rat(lexeme).sqrt()
    assert r.payload == expected
    assert r.exact is True


@pytest.mark.parametrize("lexeme", ["2", "1.5", "0.5"])  # 2/1, 3/2, 1/2 — not both-square
def test_sqrt_rational_refuses_an_irrational_root(lexeme):
    with pytest.raises(NotRepresentableError, match="irrational"):
        _rat(lexeme).sqrt()


# --- _pi_scaled(): the internal high-precision pi helper (28.10.1) ----------
# An engine primitive (Machin's formula on scaled ints), unreachable through the
# language, so it is unit-tested directly. floor(pi * 10**decimals) must be correct
# to its last place across a range of scales.

# pi to 50 places (floor): 3.14159265358979323846264338327950288419716939937510
_PI_50 = 314159265358979323846264338327950288419716939937510


def test_pi_scaled_matches_known_digits():
    assert _pi_scaled(0) == 3
    assert _pi_scaled(10) == 31415926535
    assert _pi_scaled(50) == _PI_50
    # Every shorter request is the longer one floored, digit for digit.
    for decimals in range(0, 50):
        assert _pi_scaled(decimals) == _PI_50 // 10 ** (50 - decimals)


# --- _ln10_scaled() / _ln2_scaled(): the internal ln constants (28.17.2) ----
# Engine primitives (the atanh series at unit-fraction arguments), unreachable
# through the language — ln(2) is consumed only by log2 (28.19) — so they are
# unit-tested directly against published digits. floor(ln(b) * 10**decimals) must
# be correct to its last place across a range of scales, like _pi_scaled.

# ln(10) to 50 places (floor): 2.30258509299404568401799145468436420760110148862877
_LN10_50 = 230258509299404568401799145468436420760110148862877
# ln(2) to 50 places (floor):  0.69314718055994530941723212145817656807550013436025
_LN2_50 = 69314718055994530941723212145817656807550013436025


def test_ln_constants_match_known_digits():
    assert _ln10_scaled(0) == 2
    assert _ln10_scaled(50) == _LN10_50
    assert _ln2_scaled(0) == 0
    assert _ln2_scaled(50) == _LN2_50
    # Every shorter request is the longer one floored, digit for digit.
    for decimals in range(0, 50):
        assert _ln10_scaled(decimals) == _LN10_50 // 10 ** (50 - decimals)
        assert _ln2_scaled(decimals) == _LN2_50 // 10 ** (50 - decimals)


# --- _fp_ln(): the natural-log core, base-10 reduce + atanh series (28.17) ---
# Returns (ln_total, working) with ln_total / 10**working ~ ln(x). Checked against
# math.log to ~working precision across a spread that exercises the reduction:
# x >= sqrt(10), x in [sqrt(.1), sqrt(10)), x < 1 (negative n), and x far from 1.


@pytest.mark.parametrize(
    ("mantissa", "decimals"),
    [
        (2, 0),  # ln(2), the bare-integer path
        (2_000_000, 6),  # ln(2.0) at scale 6
        (10_000_000, 6),  # ln(10.0) — reduces to r = 1, n = 1
        (500_000, 6),  # ln(0.5) — n = -1
        (1_000, 6),  # ln(0.001) — n = -3, several decades down
        (1_234_567_890, 6),  # ln(1234.56789) — n = 3
        (31_622_777, 7),  # ~sqrt(10), the upper reduction boundary
    ],
)
def test_fp_ln_matches_math_log(mantissa, decimals):
    ln_total, working = _fp_ln(FixedPoint(mantissa, decimals))
    approx = ln_total / 10**working
    assert math.isclose(approx, math.log(mantissa / 10**decimals), abs_tol=1e-9)


# --- abs_(): exact in every mode (19.5.1) ----------------------------------
# Magnitude only drops a sign, so it is representable in every mode and never
# rounds. Negatives are built with .neg() because literals are unsigned (the
# sign is a UnaryOp, not part of the lexeme).


def test_abs_floating_point():
    assert _f64("3.5").neg().abs_().payload == 3.5
    assert _f64("3.5").abs_().payload == 3.5
    assert _f64("0").abs_().payload == 0.0


def test_abs_rational():
    assert _rat("3.5").neg().abs_().payload == Fraction(7, 2)
    assert _rat("2").abs_().payload == Fraction(2)


def test_abs_fixed_point_preserves_scale():
    negative = _fp("1.50").neg()
    assert negative.abs_().payload == FixedPoint(150, 2)
    assert _fp("1.50").abs_().payload == FixedPoint(150, 2)
    assert negative.abs_().precision() == negative.precision()  # scale untouched


def test_abs_carries_exactness_through_unchanged():
    # abs introduces no rounding: the operand's flag is carried, never forced True.
    for mode, payload in (
        (Mode.FLOATING_POINT, -3.5),
        (Mode.RATIONAL, Fraction(-7, 2)),
        (Mode.FIXED_POINT, FixedPoint(-150, 2)),
    ):
        assert Value(mode, payload, exact=True).abs_().exact is True
        assert Value(mode, payload, exact=False).abs_().exact is False


def test_abs_returns_new_value_preserving_mode():
    v = _f64("3.5").neg()
    r = v.abs_()
    assert r.mode is Mode.FLOATING_POINT
    assert r is not v


# --- from_real / to_float: the solver's bridge to/from a float (31.7) --------


def test_from_real_floating_point_is_the_nearest_double():
    value = Value.from_real(0.5, Mode.FLOATING_POINT, scale=4)  # scale ignored for float
    assert value.mode is Mode.FLOATING_POINT
    assert value.payload == 0.5
    assert value.exact is False


def test_from_real_fixed_point_quantises_to_scale_and_is_exact():
    value = Value.from_real(1.25, Mode.FIXED_POINT, scale=2)
    assert value.payload == FixedPoint(125, 2)
    assert value.exact is True


def test_from_real_fixed_point_rounds_half_to_even():
    # 0.125 at scale 2 -> 0.12 (round half to even, the engine-wide rule).
    assert Value.from_real(0.125, Mode.FIXED_POINT, scale=2).payload == FixedPoint(12, 2)


def test_from_real_rational_is_the_quantised_fraction():
    value = Value.from_real(0.5, Mode.RATIONAL, scale=12)
    assert value.payload == Fraction(1, 2)
    assert value.exact is True


def test_from_real_accepts_a_fraction_input():
    value = Value.from_real(Fraction(3, 4), Mode.FIXED_POINT, scale=2)
    assert value.payload == FixedPoint(75, 2)


@pytest.mark.parametrize(
    "value, expected",
    [
        (Value(Mode.FLOATING_POINT, 1.5, exact=False), 1.5),
        (Value(Mode.FIXED_POINT, FixedPoint(125, 2), exact=True), 1.25),
        (Value(Mode.RATIONAL, Fraction(1, 4), exact=True), 0.25),
    ],
)
def test_to_float_reduces_each_mode(value, expected):
    assert value.to_float() == expected


def test_from_real_then_to_float_round_trips_on_the_grid():
    # A value already on the fixed-point grid survives the round trip exactly.
    assert Value.from_real(3.14, Mode.FIXED_POINT, scale=2).to_float() == 3.14


# --- inexact-handling enum (35.2) -----------------------------------------
# The caller-supplied policy for an inexact result, resolved by name/alias like
# Mode, with its own per-member explanation behind the abort diagnostic.


def test_inexact_handling_members_and_values():
    assert InexactHandling.CONTINUE_AND_REPORT.value == "continue-and-report"
    assert InexactHandling.ABORT_ON_INEXACT.value == "abort-on-inexact"


def test_resolve_inexact_handling_canonical_and_aliases():
    assert resolve_inexact_handling("continue-and-report") is InexactHandling.CONTINUE_AND_REPORT
    assert resolve_inexact_handling("report") is InexactHandling.CONTINUE_AND_REPORT
    assert resolve_inexact_handling("abort-on-inexact") is InexactHandling.ABORT_ON_INEXACT
    assert resolve_inexact_handling("abort") is InexactHandling.ABORT_ON_INEXACT
    assert resolve_inexact_handling("strict") is InexactHandling.ABORT_ON_INEXACT


def test_resolve_inexact_handling_unknown_raises():
    with pytest.raises(ValueError):
        resolve_inexact_handling("whenever")


def test_inexact_handling_help_covers_every_member():
    # The single-source guard: no member may ship without a help description.
    assert set(INEXACT_HANDLING_HELP) == set(InexactHandling)


def test_explain_inexact_fixed_point_rounding_reports_the_residual():
    # An algebraic fixed-point rounding carries the exact residual (35.1.2), and
    # the explanation surfaces it plus the precision / rational steer.
    rounded = Value(Mode.FIXED_POINT, FixedPoint(33, 2), exact=False, error=Fraction(-1, 300))
    text = rounded.explain_inexact()
    assert "-1/300" in text  # the exact residual
    assert "rational mode is exact" in text  # the one CERTAIN fix
    assert "min_fixed_point_precision" not in text  # never promised — may not help


def test_explain_inexact_fixed_point_irrational_has_no_residual():
    # An irrational fixed-point result (no recorded error) is named inexact in every
    # type, with no error figure and no promised fix.
    irrational = Value(Mode.FIXED_POINT, FixedPoint(141, 2), exact=False)
    text = irrational.explain_inexact()
    assert "irrational" in text
    assert "no exact value" in text


def test_explain_inexact_floating_point_steers_off_float():
    text = Value(Mode.FLOATING_POINT, 1.5, exact=False).explain_inexact()
    assert "floating-point" in text and "fixed-point or rational" in text
