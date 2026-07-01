# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Functional tests for the abacus call functions (TODO 22).

Each test exercises ONE function — abs or sqrt — through the in-process
`calculate` tool across the three modes and a spread of arguments, asserting the
tool's annotated `value` string (e.g. "16 (exact)") and, for the domain refusals,
the line-tagged error. Driving the real tool (full dispatch -> parse -> evaluate
-> render) keeps this a functional test, not a unit test of the methods.

Under `pytest -v` (run-tests.sh --verbose) conftest's _compact_functions_trace
prints every "expression [mode] = value" pair it sees on the tool seam.
"""

import asyncio
import json
from fractions import Fraction

import pytest

from mcp_abacus.expr.parser import parse
from mcp_abacus.expr.value import FixedPoint, Mode
from mcp_abacus.server import mcp


def _calc(expression, mode=None, floor=None):
    """Invoke `calculate` in-process; return its structured payload dict (25.1).

    `mode` is omitted from the request when None, exercising the fixed-point default.
    `floor` (min_fixed_point_precision) is likewise omitted when None — supplied only
    by the nullary-precision tests (29.4) that raise the derived scale past the
    literals' own.
    """
    arguments = {"expression": expression}
    if mode is not None:
        arguments["mode"] = mode
    if floor is not None:
        arguments["min_fixed_point_precision"] = floor
    result = asyncio.run(mcp.call_tool("calculate", arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def _value(expression, mode=None, floor=None):
    """The annotated `value` string on success; fail loudly if the call errored."""
    payload = _calc(expression, mode, floor)
    assert payload["error"] is None, f"{expression!r} [{mode}] errored: {payload['error']}"
    return payload["value"]


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # abs is exact wherever the mode has an exact grid — magnitude only drops a
        # sign. Negatives reach abs through unary minus or a subexpression.
        ("abs(-3)", None, "3 (exact)"),  # fixed-point default
        ("abs(3)", None, "3 (exact)"),
        ("abs(2 - 5)", None, "3 (exact)"),  # negative via a subexpression
        ("abs(-2.50)", None, "2.50 (exact)"),  # scale preserved
        ("abs(0)", None, "0 (exact)"),
        ("abs(-3)", "rational", "3 (exact)"),
        ("abs(-1/3)", "rational", "1/3 (exact)"),
        # binary64 carries its inexact flag through abs (the literal is already inexact).
        ("abs(-3)", "floating-point", "3.0 (inexact)"),
        ("abs(-2.5)", "floating-point", "2.5 (inexact)"),
    ],
)
def test_abs(expression, mode, value):
    assert _value(expression, mode) == value


# --- sum(i, lo, hi, expr): the range-summation (Σ) special form (40.19) --------
# A SOLVER-ADJACENT special form like integral/diff: `i` is a bare NAME (the index) and
# `expr` the unevaluated body, NOT values. A FINITE EXACT fold over INTEGER steps from lo to
# hi inclusive — repeated +, so (unlike integral/diff) it is EXACT in every mode and inherits
# the body's own exactness, never stamping inexact itself.
@pytest.mark.parametrize(
    ("expression", "mode", "floor", "value"),
    [
        ("sum(i, 1, 4, i)", None, None, "10 (exact)"),  # 1+2+3+4, fixed-point default
        ("sum(i, 1, 4, i**2)", None, None, "30 (exact)"),  # 1+4+9+16
        ("sum(i, 1, 3, 2*i)", None, None, "12 (exact)"),  # the body is an expression in i
        # The body reads the index at the precision FLOOR, so a raised floor carries through.
        ("sum(i, 1, 3, i)", None, 4, "6.0000 (exact)"),
        # An EMPTY range (lo > hi) is the additive identity 0, a whole number at scale 0.
        ("sum(i, 3, 1, i)", None, None, "0 (exact)"),
        ("sum(i, 5, 5, i**2)", None, None, "25 (exact)"),  # single-term range
        # rational folds exactly — the harmonic partial sum 1 + 1/2 + 1/3 + 1/4.
        ("sum(i, 1, 4, 1/i)", "rational", None, "25/12 (exact)"),
        # binary64 carries the inexact flag through, like every float op.
        ("sum(i, 1, 3, i)", "floating-point", None, "6.0 (inexact)"),
    ],
)
def test_sum(expression, mode, floor, value):
    assert _value(expression, mode, floor) == value


def test_sum_inherits_body_inexactness():
    # An inexact term (fixed-point sqrt rounds) makes the whole sum inexact, exactly as
    # repeated + would — the fold propagates the flag, it never stamps inexact itself.
    assert _value("sum(i, 1, 2, sqrt(i))").startswith("2 (inexact")


def test_sum_reads_outer_variables():
    # The body re-evaluates in a child store seeded from the run's, so it can read an outer
    # binding (k) while the index (i) shadows the per-term value. k*(1+2+3) = 3*6 = 18.
    assert _value("k = 3\nsum(i, 1, 3, k*i)", "rational") == "18 (exact)"


def test_sum_nests_with_the_outer_index_feeding_the_inner_bound():
    # A special form nested in another's body: the inner sum's upper bound is the outer
    # index i, so the body re-evaluates with i rebound each outer term —
    # Σ_{i=1..3} Σ_{j=1..i} j = 1 + 3 + 6 = 10. Pins the child-store seeding across forms.
    assert _value("sum(i, 1, 3, sum(j, 1, i, j))") == "10 (exact)"


def test_inner_index_shadows_an_outer_index_of_the_same_name():
    # When the nested fold reuses the OUTER index name, the inner binding shadows it (it is
    # not 1+2 doubled): Σ_{i=1..2} (Σ_{i=1..2} i) = Σ_{i=1..2} 3 = 6. Proves each form's
    # index is a fresh dummy in its own child scope, not a shared outer variable.
    assert _value("sum(i, 1, 2, sum(i, 1, 2, i))") == "6 (exact)"


def test_sum_masks_its_index_in_referenced_names():
    # The index is BOUND (a dummy the fold rebinds per term), so it is not a free reference —
    # only genuinely free names (here `n` and `t`) leak out, keeping the index from being
    # mistaken for a solver unknown when a sum is nested. (The index is the FIRST arg here,
    # unlike integral/diff where it is the second — _SPECIAL_FORM_BOUND_VAR tracks which.)
    assert parse("sum(i, 1, n, i*t)").referenced_names() == frozenset({"n", "t"})


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # max/min SELECT an operand — never compute — so the result is that operand
        # verbatim: its own scale, no covering, no rounding.
        ("max(1, 2, 3)", None, "3 (exact)"),  # fixed-point default
        ("min(1, 2, 3)", None, "1 (exact)"),
        ("max(5)", None, "5 (exact)"),  # single arg -> identity
        ("min(5)", None, "5 (exact)"),
        ("max(-3, -5)", None, "-3 (exact)"),  # ordering respects sign
        ("min(-3, -5)", None, "-5 (exact)"),
        # The chosen operand keeps its OWN scale — no covering to a common one.
        ("max(1.5, 2.250)", None, "2.250 (exact)"),  # scale 3 preserved
        ("min(1.5, 2.250)", None, "1.5 (exact)"),  # scale 1 preserved
        # A tie keeps the EARLIER operand (so its scale wins) — stable selection.
        ("max(1.5, 1.50)", None, "1.5 (exact)"),
        ("min(1.50, 1.5)", None, "1.50 (exact)"),
        # rational compares exact fractions.
        ("max(1/2, 2/3, 1/6)", "rational", "2/3 (exact)"),
        ("min(1/2, 2/3, 1/6)", "rational", "1/6 (exact)"),
        # binary64 selects too, carrying the operand's inexact flag.
        ("max(1, 2, 3)", "floating-point", "3.0 (inexact)"),
        ("min(1.5, 2.5)", "floating-point", "1.5 (inexact)"),
        # A SINGLE vector operand is reduced over its ELEMENTS (19.1.10) — the same
        # verbatim selection, just over the list instead of the arguments.
        ("max([3, 1, 2])", None, "3 (exact)"),  # fixed-point default
        ("min([3, 1, 2])", None, "1 (exact)"),
        ("max([5])", None, "5 (exact)"),  # one element -> that element
        ("min([5])", None, "5 (exact)"),
        ("min([-3, -5, -1])", None, "-5 (exact)"),  # ordering respects sign
        # The chosen element keeps its OWN scale, and a tie keeps the EARLIEST.
        ("max([1.5, 2.250, 0.5])", None, "2.250 (exact)"),  # scale 3 preserved
        ("min([1.50, 1.5, 1.500])", None, "1.50 (exact)"),  # tie -> earliest's scale
        # rational / binary64 elements reduce exactly the way scalar args do.
        ("max([1/2, 2/3, 1/6])", "rational", "2/3 (exact)"),
        ("min([1, 2, 3])", "floating-point", "1.0 (inexact)"),
    ],
)
def test_max_min(expression, mode, value):
    assert _value(expression, mode) == value


def test_max_min_carry_the_chosen_operands_exactness():
    # Selection carries the picked operand's own exactness: max keeps the exact 2,
    # min picks the inexact sqrt(2) (which rounds in fixed-point) and is inexact.
    # A vector reduction selects verbatim too, so it carries the element's flag.
    assert _value("max(2, sqrt(2))") == "2 (exact)"
    assert _value("min(2, sqrt(2))").startswith("1 (inexact")
    assert _value("max([2, sqrt(2)])") == "2 (exact)"
    assert _value("min([2, sqrt(2)])").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        # A vector is legal only as the SOLE operand (19.1.10): the two forms are an
        # OVERLOAD, not a blend, so mixing a vector with a scalar or another vector
        # refuses with a message spelling both forms out (defers 40.13).
        ("min([1, 2], 3)", "min has two forms — min(vector) or min(a, b, …) — and cannot mix them"),
        ("max(3, [1, 2])", "max has two forms — max(vector) or max(a, b, …) — and cannot mix them"),
        (
            "min([1, 2], [3, 4])",
            "min has two forms — min(vector) or min(a, b, …) — and cannot mix them",
        ),
        # An empty vector has nothing to select from — a minimum/maximum of no values.
        ("min([])", "min of an empty vector is undefined"),
        ("max([])", "max of an empty vector is undefined"),
    ],
)
def test_max_min_vector_refusals(expression, error):
    payload = _calc(expression)
    assert payload["value"] is None, f"{expression!r} unexpectedly succeeded"
    assert payload["error"] == error


# --- product(i, lo, hi, expr): the range-product (Π) special form (40.19) ------
# sum's multiplicative twin (40.19): the SAME finite fold over INTEGER steps, but repeated *
# instead of +, so unlike sum it COMPUTES — fixed-point may round to the covering scale,
# rational stays exact, float rounds. Still EXACT-or-the-body's-inexactness; the fold never
# stamps inexact itself, the rounding is the multiply's.
@pytest.mark.parametrize(
    ("expression", "mode", "floor", "value"),
    [
        ("product(i, 1, 4, i)", None, None, "24 (exact)"),  # 4! = 1*2*3*4
        ("product(i, 2, 4, i)", None, None, "24 (exact)"),  # 2*3*4
        # An EMPTY range (lo > hi) is the multiplicative identity 1, a whole number at scale 0.
        ("product(i, 3, 1, i)", None, None, "1 (exact)"),
        ("product(i, 5, 5, i)", None, None, "5 (exact)"),  # single-term range
        # The covering scale rounds: 1.5 * 1.5 = 2.25 but scale 1 -> rounds, flagged inexact.
        (
            "product(i, 1, 2, 1.5)",
            None,
            None,
            "2.2 (inexact, rounded to 1 decimal — pass min_fixed_point_precision "
            "for more; e.g. =5 → 2.25000)",
        ),
        # rational multiplies exactly — 1/1 * 1/2 * 1/3 * 1/4 = 1/24.
        ("product(i, 1, 4, 1/i)", "rational", None, "1/24 (exact)"),
        # binary64 carries the inexact flag through (like every float op).
        ("product(i, 1, 3, i)", "floating-point", None, "6.0 (inexact)"),
    ],
)
def test_product(expression, mode, floor, value):
    assert _value(expression, mode, floor) == value


def test_product_inherits_body_inexactness():
    # An inexact term (fixed-point sqrt rounds) makes the whole product inexact, exactly as
    # repeated * would — the fold propagates the flag.
    assert _value("product(i, 1, 2, sqrt(i))").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # The index (1st arg) must be a bare name: a literal or a constant nullary (pi/e) is
        # not — the same stance integral/diff take on their variable argument.
        ("sum(5, 1, 3, 5)", None, "sum's index (1st argument) must be a name"),
        ("product(pi, 1, 3, 2)", None, "product's index (1st argument) must be a name"),
        # The bounds must be integers — the fold steps by one integer at a time, so a
        # fractional bound REFUSES through the same _as_integer gate gcd/lcm/factorial use.
        ("sum(i, 1.5, 3, i)", None, "sum bounds requires integer operands"),
        ("product(i, 1, 3.5, i)", None, "product bounds requires integer operands"),
        # A range wider than the term cap REFUSES rather than grinding the server — the
        # DoS guard (factorial's cap sized far higher for series sums).
        ("sum(i, 1, 200000, i)", None, "sum range exceeds 100000 terms (200000 requested)"),
        # A transcendental body in rational mode refuses at the term, like sin everywhere —
        # surfaced as the body's own line-tagged error (i=1 is a non-zero rational).
        ("sum(i, 1, 3, sin(i))", "rational", "sine of a non-zero rational is irrational"),
    ],
)
def test_range_fold_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_range_fold_arity_is_a_parse_error():
    # Fixed arity 4, wired through FUNCTION_ARITIES like every call — wrong count is caught at
    # parse, before evaluation, the same as a misused ordinary function.
    payload = _calc("sum(i, 1, 3)")
    assert payload["value"] is None
    assert "takes 4 argument(s)" in payload["error"]


# --- map(vector, name, body): the element-wise transform special form (40.24) --
# A higher-order special form like sum/product (40.19): `name` is a bare NAME (the element
# dummy) and `body` the unevaluated expression, NOT values — but the 1st arg is a VECTOR that
# IS evaluated, and the result is a VECTOR (not a scalar). The FIRST form that both consumes
# and produces a vector. map only dispatches: exactness is the body's, per element.
@pytest.mark.parametrize(
    ("expression", "mode", "floor", "value"),
    [
        ("map([1, 2, 3], x, x**2)", None, None, "[1, 4, 9] (exact)"),  # fixed-point default
        ("map([1, 2, 3], x, 2*x + 1)", None, None, "[3, 5, 7] (exact)"),  # body is an expr in x
        # An EMPTY source maps to an empty vector, vacuously exact in every mode.
        ("map([], x, x**2)", None, None, "[] (exact)"),
        # The result carries the run's element mode: rational holds the reciprocals exactly.
        ("map([1, 2, 4], x, 1/x)", "rational", None, "[1, 1/2, 1/4] (exact)"),
        # A float body makes every element — and so the vector — inexact.
        ("map([1.0, 2.0], x, x*2)", "floating-point", None, "[2.0, 4.0] (inexact)"),
        # The body reads the element at the run scale, so a raised floor carries through.
        ("map([1, 2], x, x)", None, 4, "[1.0000, 2.0000] (exact)"),
    ],
)
def test_map(expression, mode, floor, value):
    assert _value(expression, mode, floor) == value


def test_map_inherits_body_inexactness():
    # An inexact element (fixed-point sqrt rounds) makes the whole vector inexact, exactly as
    # a vector literal folds its elements' flags — map stamps no verdict of its own.
    assert _value("map([2, 3], x, sqrt(x))").startswith("[1, 2] (inexact")


def test_map_consumes_a_produced_vector():
    # map both CONSUMES and PRODUCES a vector, so it composes with the other vector-producer:
    # factor(12) = [2, 2, 3], squared element-wise is [4, 4, 9].
    assert _value("map(factor(12), x, x**2)") == "[4, 4, 9] (exact)"


def test_map_reads_outer_variables():
    # The body re-evaluates in a child store seeded from the run's, so it reads an outer
    # binding (k) while the element dummy (x) shadows the per-element value: k*x over [1,2,3].
    assert _value("k = 10\nmap([1, 2, 3], x, k*x)") == "[10, 20, 30] (exact)"


def test_map_element_dummy_shadows_an_outer_binding_of_the_same_name():
    # When the element name reuses an OUTER binding, the per-element value shadows it inside
    # the body: x is 1,2,3 in turn, not the outer 99. Proves the dummy is a fresh child scope.
    assert _value("x = 99\nmap([1, 2, 3], x, x + 1)") == "[2, 3, 4] (exact)"


def test_map_masks_its_element_name_in_referenced_names():
    # The element name is BOUND (a dummy map rebinds per element), so it is not a free
    # reference — only genuinely free names (here the source `data` and the body's `t`) leak
    # out, keeping the dummy from being mistaken for a solver unknown. (The name is the SECOND
    # arg, like integral/diff's variable — _SPECIAL_FORM_BOUND_VAR tracks which.)
    assert parse("map(data, x, x*t)").referenced_names() == frozenset({"data", "t"})


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # The 1st arg must be a VECTOR: map transforms a series, so a scalar has no elements to
        # walk and refuses rather than silently wrapping in a 1-element vector.
        ("map(5, x, x**2)", None, "map's first argument must be a vector"),
        # The element name (2nd arg) must be a bare name: a literal or a constant nullary
        # (pi/e) is not — the same stance sum/integral/diff take on their variable argument.
        ("map([1, 2], 3, x)", None, "map's variable (2nd argument) must be a name"),
        ("map([1, 2], pi, x)", None, "map's variable (2nd argument) must be a name"),
        # A transcendental body in rational mode refuses at the element, like sin everywhere —
        # surfaced as the body's own line-tagged error (the element 1 is a non-zero rational).
        ("map([1, 2], x, sin(x))", "rational", "sine of a non-zero rational is irrational"),
        # A body that itself produces a vector would nest — refused by the 1-D vector rule.
        (
            "map([1, 2], x, [x, x])",
            None,
            "vectors must be one-dimensional; a vector cannot hold a vector",
        ),
    ],
)
def test_map_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_map_arity_is_a_parse_error():
    # Fixed arity 3, wired through FUNCTION_ARITIES like every call — wrong count is caught at
    # parse, before evaluation, the same as a misused ordinary function.
    payload = _calc("map([1, 2, 3], x)")
    assert payload["value"] is None
    assert "takes 3 argument(s)" in payload["error"]


# --- residual_sum_squares(expr, var, xs, ys): the least-squares cost special form (40.25) ---
# A higher-order special form like map/sum: `var` is a bare NAME (the model's free variable)
# and `expr` the unevaluated model, NOT values — but args 3/4 are two EQUAL-LENGTH data vectors
# walked in LOCKSTEP. It folds Σ(ys_i - expr[var:=xs_i])**2 to a SCALAR: like sum it stamps no
# verdict, so exactness is the subtract+square+add arithmetic's (exact in rational, may round in
# fixed-point/float).
@pytest.mark.parametrize(
    ("expression", "mode", "floor", "value"),
    [
        # A perfect fit leaves zero residual: model 2*x hits [2,4,6] exactly on x=[1,2,3].
        ("residual_sum_squares(2*x, x, [1, 2, 3], [2, 4, 6])", None, None, "0 (exact)"),
        # residuals [0, 0, 1] -> 0 + 0 + 1 = 1 (integer data stays exact in fixed-point).
        ("residual_sum_squares(2*x, x, [1, 2, 3], [2, 4, 7])", None, None, "1 (exact)"),
        # The model is any expression in the variable: x**2 vs [1, 4, 9] is a perfect fit.
        ("residual_sum_squares(x**2, x, [1, 2, 3], [1, 4, 9])", None, None, "0 (exact)"),
        # rational squares exactly: gaps 1/2, 1/2 -> 1/4 + 1/4 = 1/2.
        ("residual_sum_squares(x, x, [1, 2], [3/2, 5/2])", "rational", None, "1/2 (exact)"),
        # binary64 carries the inexact flag through: gaps 1, 2 -> 1 + 4 = 5.
        (
            "residual_sum_squares(x, x, [1.0, 2.0], [2.0, 4.0])",
            "floating-point",
            None,
            "5.0 (inexact)",
        ),
    ],
)
def test_residual_sum_squares(expression, mode, floor, value):
    assert _value(expression, mode, floor) == value


def test_residual_sum_squares_reads_outer_variables():
    # The model re-evaluates in a child store seeded from the run's, so it reads an outer
    # binding (a) while the variable (x) shadows the per-point value. With a=2 the model 2*x
    # fits [2,4,6] perfectly, so the cost is 0.
    assert _value("a = 2\nresidual_sum_squares(a*x, x, [1, 2, 3], [2, 4, 6])", "rational") == (
        "0 (exact)"
    )


def test_residual_sum_squares_masks_its_variable_in_referenced_names():
    # The model variable is BOUND (a dummy the fold rebinds per point), so it is not a free
    # reference — only genuinely free names (the model's `a`/`b` and the data `xs`/`ys`) leak
    # out, keeping the variable from being mistaken for a solver unknown. (The name is the
    # SECOND arg, like integral/diff/map — _SPECIAL_FORM_BOUND_VAR tracks which.)
    names = parse("residual_sum_squares(a*x + b, x, xs, ys)").referenced_names()
    assert names == frozenset({"a", "b", "xs", "ys"})


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # The variable (2nd arg) must be a bare name: a literal or a constant nullary (pi/e) is
        # not — the same stance the other special forms take on their variable argument.
        (
            "residual_sum_squares(2*x, 3, [1, 2], [1, 2])",
            None,
            "residual_sum_squares's variable (2nd argument) must be a name",
        ),
        # The data (3rd/4th args) must both be vectors — a scalar has no points to pair.
        (
            "residual_sum_squares(2*x, x, 5, [1, 2])",
            None,
            "residual_sum_squares's data (3rd and 4th arguments) must be two vectors",
        ),
        # Unequal-length data is unpaired and REFUSES, mirroring covariance.
        (
            "residual_sum_squares(2*x, x, [1, 2, 3], [1, 2])",
            None,
            "residual_sum_squares requires two equal-length data vectors",
        ),
        # Empty data is undefined (no points to fit) and REFUSES.
        (
            "residual_sum_squares(2*x, x, [], [])",
            None,
            "residual_sum_squares of empty data is undefined",
        ),
        # A transcendental model in rational mode refuses at the point, like sin everywhere —
        # surfaced as the model's own line-tagged error (the point 1 is a non-zero rational).
        (
            "residual_sum_squares(sin(x), x, [1, 2], [1, 2])",
            "rational",
            "sine of a non-zero rational is irrational",
        ),
    ],
)
def test_residual_sum_squares_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_residual_sum_squares_arity_is_a_parse_error():
    # Fixed arity 4, wired through FUNCTION_ARITIES like every call — wrong count is caught at
    # parse, before evaluation, the same as a misused ordinary function.
    payload = _calc("residual_sum_squares(2*x, x, [1, 2])")
    assert payload["value"] is None
    assert "takes 4 argument(s)" in payload["error"]


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # avg COMPUTES (sum / count), so it follows the mode's / rule: fixed-point
        # quantizes to the covering scale, rational is exact, float rounds.
        ("avg(2, 4, 6)", None, "4 (exact)"),  # fixed-point default, divides evenly
        # (1 + 2) / 2 = 1.5 but the covering scale is 0 -> rounds, flagged inexact.
        (
            "avg(1, 2)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5000)",
        ),
        ("avg(5)", None, "5 (exact)"),  # single arg -> a / 1, the identity
        ("avg(1.0, 2.0)", None, "1.5 (exact)"),  # scale 1 covers the halving
        ("avg(1, 2, 3)", "rational", "2 (exact)"),  # rational divides exactly
        ("avg(1, 2)", "rational", "3/2 (exact)"),  # ... even when it does not reduce
        # binary64 rounds and carries the inexact flag.
        ("avg(2, 4)", "floating-point", "3.0 (inexact)"),
        # A SINGLE vector is reduced over its ELEMENTS (19.1.10) — same result as the
        # equivalent flat run, in every mode.
        ("avg([2, 4, 6])", None, "4 (exact)"),
        ("avg([1, 2])", "rational", "3/2 (exact)"),
    ],
)
def test_avg(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # median orders by value (no arithmetic) and, for an ODD count, SELECTS the
        # middle operand verbatim — exact, carrying its own scale, like max/min.
        ("median(3, 1, 2)", None, "2 (exact)"),  # fixed-point default, unsorted input
        ("median(5)", None, "5 (exact)"),  # single arg -> identity
        ("median(-5, -1, -3)", None, "-3 (exact)"),  # ordering respects sign
        ("median(1/2, 1/6, 2/3)", "rational", "1/2 (exact)"),  # exact fractions
        ("median(1, 2, 3)", "floating-point", "2.0 (inexact)"),  # selects, carries flag
        # An EVEN count averages the two middles -> follows the / rule, like avg.
        ("median(1, 3)", None, "2 (exact)"),  # (1 + 3) / 2 = 2 fits scale 0 exactly
        # (1 + 2) / 2 of the two middles of 1,2,3,4 = 1.5 -> rounds at scale 0.
        (
            "median(1, 2, 3, 4)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.5000)",
        ),
        ("median(2, 4)", "rational", "3 (exact)"),  # even average, exact in rational
        # A SINGLE vector is reduced over its ELEMENTS (19.1.10) — odd selects, even
        # averages, just as the flat forms.
        ("median([3, 1, 2])", None, "2 (exact)"),
        ("median([2, 4])", "rational", "3 (exact)"),
    ],
)
def test_median(expression, mode, value):
    assert _value(expression, mode) == value


def test_median_odd_carries_the_selected_operands_exactness():
    # An odd-count median SELECTS the middle operand verbatim, so it carries that
    # operand's own exactness — here the inexact (rounded) fixed-point sqrt(2).
    assert _value("median(1, sqrt(2), 3)").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # quantile(q in [0, 1], data…) — the type-7 order statistic generalising
        # median (28.7). A rank landing ON a datum returns it VERBATIM (exact like
        # median's odd case): (n-1)*q integral selects sorted[rank].
        ("quantile(0, 3, 1, 2)", None, "1 (exact)"),  # min
        ("quantile(1, 3, 1, 2)", None, "3 (exact)"),  # max
        ("quantile(0.25, 1, 2, 3, 4, 5)", None, "2 (exact)"),  # (5-1)*0.25 = 1 -> sorted[1]
        ("quantile(0.5, 1, 2, 3)", None, "2 (exact)"),  # odd count, lands on the middle
        ("quantile(0.5, 5)", None, "5 (exact)"),  # a lone datum is every quantile
        # Data is sorted by value first, so order of arguments does not matter.
        ("quantile(0.5, 4, 2, 3, 1)", "rational", "5/2 (exact)"),
        # A fractional rank INTERPOLATES linearly between the two straddling data, so it
        # follows the mode's / rule exactly as median's even case: rational EXACT...
        ("quantile(0.5, 1, 2, 3, 4)", "rational", "5/2 (exact)"),  # midway 2..3
        ("quantile(0.3, 1, 2, 3, 4)", "rational", "19/10 (exact)"),  # 1 + 0.9*(2-1)
        ("quantile(0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)", "rational", "19/10 (exact)"),
        # ... float rounds and flags inexact...
        ("quantile(0.5, 1, 2, 3, 4)", "floating-point", "2.5 (inexact)"),
        # ... and fixed-point at scale 0 rounds 2.5 with the precision hint (matching
        # median(1, 2, 3, 4)); a wider floor recovers it exactly.
        (
            "quantile(0.5, 1, 2, 3, 4)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.5000)",
        ),
        # A SINGLE vector is the data (19.1.10) — same as the flat run.
        ("quantile(0.5, [4, 2, 3, 1])", "rational", "5/2 (exact)"),
        ("quantile(0.25, [1, 2, 3, 4, 5])", None, "2 (exact)"),
    ],
)
def test_quantile(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # percentile(p in [0, 100], data…) is quantile scaled by 100, so percentile(50,
        # …) is the median (28.7) and agrees datum-for-datum with quantile(0.5, …).
        ("percentile(50, 1, 2, 3)", None, "2 (exact)"),
        ("percentile(25, 1, 2, 3, 4, 5)", None, "2 (exact)"),
        ("percentile(50, 1, 2, 3, 4)", "rational", "5/2 (exact)"),
        ("percentile(50, 1, 2, 3, 4)", "floating-point", "2.5 (inexact)"),
        (
            "percentile(50, 1, 2, 3, 4)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.5000)",
        ),
        ("percentile(50, [1, 2, 3])", None, "2 (exact)"),
    ],
)
def test_percentile(expression, mode, value):
    assert _value(expression, mode) == value


def test_quantile_widening_the_floor_recovers_the_interpolated_value():
    # The interpolation rounds only where median's even case does; raising the
    # fixed-point floor recovers it exactly, like median (29.4 / 28.7).
    assert _value("quantile(0.5, 1, 2, 3, 4)", floor=4) == "2.5000 (exact)"
    assert _value("percentile(50, 1, 2, 3, 4)", floor=4) == "2.5000 (exact)"


def test_quantile_landing_on_a_datum_carries_its_exactness():
    # A rank that lands on a datum SELECTS it verbatim, carrying that operand's own
    # exactness — here the inexact (rounded) fixed-point sqrt(2) at the median rank.
    assert _value("quantile(0.5, 1, sqrt(2), 3)").startswith("1 (inexact")


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        # The point is a fraction/percentage of the range — outside it is undefined.
        ("quantile(7, 1, 2)", "quantile point must be between 0 and 1"),
        ("quantile(-0.1, 1, 2)", "quantile point must be between 0 and 1"),
        ("percentile(150, 1, 2)", "percentile point must be between 0 and 100"),
        ("percentile(-1, 1, 2)", "percentile point must be between 0 and 100"),
        # The data has the same two forms as median, minus the leading point.
        ("quantile(0.5, [])", "quantile of an empty vector is undefined"),
        (
            "quantile(0.5, [1, 2], 3)",
            "quantile has two data forms — quantile(p, vector) or quantile(p, a, b, …) "
            "— and cannot mix them",
        ),
        # The point must be a scalar, not a vector.
        ("quantile([0.5], 1, 2)", "quantile's point must be a scalar, not a vector"),
    ],
)
def test_quantile_refusals(expression, error):
    payload = _calc(expression)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # POPULATION variance (/ n): sum of squared deviations from the mean / n.
        # The textbook 2,4,4,4,5,5,7,9 has mean 5, squared-deviation sum 32, /8 = 4 —
        # all-integer arithmetic, so exact in every mode.
        ("variance(2, 4, 4, 4, 5, 5, 7, 9)", None, "4 (exact)"),  # fixed-point default
        ("variance(2, 4, 4, 4, 5, 5, 7, 9)", "rational", "4 (exact)"),
        ("variance(2, 4, 4, 4, 5, 5, 7, 9)", "floating-point", "4.0 (inexact)"),
        ("variance(1, 2, 3, 4, 5)", None, "2 (exact)"),  # mean 3, sum 10, /5 = 2
        ("variance(5)", None, "0 (exact)"),  # a lone point has no spread
        # rational keeps the exact mean and divide even when they do not reduce.
        ("variance(1, 2)", "rational", "1/4 (exact)"),  # mean 3/2, devs ±1/2, /2
        ("variance(2, 4, 6)", "rational", "8/3 (exact)"),  # mean 4, sum 8, /3
        # binary64 rounds; fixed-point at scale 0 rounds 8/3 and flags inexact.
        ("variance(2, 4, 6)", "floating-point", "2.6666666666666665 (inexact)"),
        (
            "variance(2, 4, 6)",
            None,
            "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.6667)",
        ),
        # A SINGLE vector is reduced over its ELEMENTS (19.1.10) — same as the flat run.
        ("variance([1, 2, 3, 4, 5])", None, "2 (exact)"),
        ("variance([2, 4, 6])", "rational", "8/3 (exact)"),
    ],
)
def test_variance(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # stddev == sqrt(variance), inheriting sqrt's per-mode story. The textbook
        # set's variance 4 has an exact root 2 in every mode (a perfect square).
        ("stddev(2, 4, 4, 4, 5, 5, 7, 9)", None, "2 (exact)"),  # fixed-point default
        ("stddev(2, 4, 4, 4, 5, 5, 7, 9)", "rational", "2 (exact)"),
        ("stddev(2, 4, 4, 4, 5, 5, 7, 9)", "floating-point", "2.0 (inexact)"),
        ("stddev(1, 2)", "rational", "1/2 (exact)"),  # variance 1/4, exact root
        ("stddev(5)", None, "0 (exact)"),  # sqrt(0) = 0
        # Irrational root: float/fixed-point round and flag inexact (rational refuses,
        # tested separately). variance(1,2,3,4,5) = 2, sqrt(2) is irrational.
        ("stddev(1, 2, 3, 4, 5)", "floating-point", "1.4142135623730951 (inexact)"),
        (
            "stddev(1, 2, 3, 4, 5)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.4142)",
        ),
        # A SINGLE vector is reduced over its ELEMENTS (19.1.10) — same as the flat run.
        ("stddev([2, 4, 4, 4, 5, 5, 7, 9])", None, "2 (exact)"),
    ],
)
def test_stddev(expression, mode, value):
    assert _value(expression, mode) == value


def test_stddev_rational_refuses_an_irrational_root():
    # In rational mode stddev inherits sqrt's exact-or-refuse pitch: variance 2 has
    # no rational root, so it raises rather than fabricate digits (line-tagged).
    payload = _calc("stddev(1, 2, 3, 4, 5)", "rational")
    assert payload["error"] == "rational square root is irrational"
    assert payload["value"] is None


def test_stddev_empty_vector_surfaces_variances_refusal():
    # stddev is sqrt(variance), so an empty vector is caught by variance and its
    # message names variance — the same composition that surfaces sqrt's message above.
    payload = _calc("stddev([])")
    assert payload["error"] == "variance of an empty vector is undefined"
    assert payload["value"] is None


@pytest.mark.parametrize(
    "expression",
    [
        # The whole computing-stats family now shares min/max's two OVERLOADS (a flat
        # run OR a single vector, 19.1.10): an empty vector has nothing to aggregate,
        # and a vector cannot be mixed with another operand. (stddev delegates to
        # variance, so its empty-vector message is variance's — see the dedicated case.)
        "avg([])",
        "median([])",
        "variance([])",
    ],
)
def test_vector_stats_refuse_an_empty_vector(expression):
    op = expression[: expression.index("(")]
    payload = _calc(expression)
    assert payload["error"] == f"{op} of an empty vector is undefined"
    assert payload["value"] is None


@pytest.mark.parametrize(
    "expression",
    [
        "avg([1, 2], 3)",
        "variance([1, 2], 3)",
        "median(3, [1, 2])",
    ],
)
def test_vector_stats_refuse_mixing_a_vector_with_operands(expression):
    op = expression[: expression.index("(")]
    payload = _calc(expression)
    assert payload["error"] == (
        f"{op} has two forms — {op}(vector) or {op}(a, b, …) — and cannot mix them"
    )
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # POPULATION covariance of two PAIRED vectors: mean((x-mx)*(y-my)). The
        # all-integer 2,4,6,8 / 1,3,5,7 has means 5 and 4, paired products 9,1,1,9 = 20,
        # /4 = 5 — exact in every mode.
        ("covariance([2, 4, 6, 8], [1, 3, 5, 7])", None, "5 (exact)"),
        ("covariance([2, 4, 6, 8], [1, 3, 5, 7])", "rational", "5 (exact)"),
        ("covariance([2, 4, 6, 8], [1, 3, 5, 7])", "floating-point", "5.0 (inexact)"),
        # Perfectly anti-correlated series gives a NEGATIVE covariance.
        ("covariance([1, 2, 3], [6, 5, 4])", "rational", "-2/3 (exact)"),
        # A lone pair has no spread -> 0, exact in every mode.
        ("covariance([5], [9])", None, "0 (exact)"),
        # It COMPUTES through the means and final divide, so it follows the mode's /
        # rule like variance: rational stays exact even when it does not reduce...
        ("covariance([1, 2, 3], [4, 5, 6])", "rational", "2/3 (exact)"),
        # ... while fixed-point at scale 0 rounds 2/3 and flags inexact, float rounds.
        ("covariance([1, 2, 3], [4, 5, 6])", "floating-point", "0.6666666666666666 (inexact)"),
        (
            "covariance([1, 2, 3], [4, 5, 6])",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.6667)",
        ),
    ],
)
def test_covariance(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        # covariance is the genuinely TWO-vector shape (40.13): both operands must be
        # vectors, equal length, and non-empty.
        ("covariance([1, 2, 3], [4, 5])", "covariance requires two equal-length vectors"),
        ("covariance([], [])", "covariance of empty vectors is undefined"),
        ("covariance([1, 2], 3)", "covariance takes two vectors — covariance(x, y)"),
        ("covariance(3, [1, 2])", "covariance takes two vectors — covariance(x, y)"),
    ],
)
def test_covariance_refusals(expression, error):
    payload = _calc(expression)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Pearson r = cov/(stddev*stddev), standardized to [-1, 1]. It INHERITS
        # stddev's sqrt, so it is ALWAYS inexact in float/fixed-point. A perfectly
        # (anti-)correlated pair is +1 / -1.
        ("correlation([1, 2, 3], [2, 4, 6])", "floating-point", "1.0 (inexact)"),
        ("correlation([1, 2, 3], [6, 4, 2])", "floating-point", "-1.0 (inexact)"),
        # A partial association lands strictly inside [-1, 1].
        (
            "correlation([1, 2, 3, 4], [1, 3, 2, 5])",
            "floating-point",
            "0.8315218406202998 (inexact)",
        ),
        # Fixed-point inherits the inexact flag too; at scale 0 the rounded cov and
        # stddevs compound badly (the hint recovers ~0.8316 at a wider scale).
        (
            "correlation([1, 2, 3, 4], [1, 3, 2, 5])",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.8316)",
        ),
    ],
)
def test_correlation(expression, mode, value):
    assert _value(expression, mode) == value


def test_correlation_rational_refuses_an_irrational_root():
    # correlation composes stddev, so it inherits sqrt's exact-or-refuse pitch in
    # rational mode: stddev([1,2,3]) = sqrt(2/3) is irrational, so it raises rather
    # than fabricate digits (line-tagged) — even for a perfectly correlated pair.
    payload = _calc("correlation([1, 2, 3], [2, 4, 6])", "rational")
    assert payload["error"] == "rational square root is irrational"
    assert payload["value"] is None


def test_correlation_of_a_constant_series_divides_by_zero():
    # A series that does not vary has zero stddev, so r divides by zero and raises —
    # Pearson correlation is undefined when a series is constant.
    payload = _calc("correlation([2, 2, 2], [1, 2, 3])")
    assert payload["error"] == "fixed-point division by zero"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        # correlation shares covariance's two-vector domain (the _paired_vectors
        # helper), but its messages name correlation.
        ("correlation([1, 2, 3], [4, 5])", "correlation requires two equal-length vectors"),
        ("correlation([], [])", "correlation of empty vectors is undefined"),
        ("correlation([1, 2], 3)", "correlation takes two vectors — correlation(x, y)"),
        ("correlation(3, [1, 2])", "correlation takes two vectors — correlation(x, y)"),
    ],
)
def test_correlation_refusals(expression, error):
    payload = _calc(expression)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Fixed-point: exact only when the root lands on the grid (a perfect square at
        # the operand's scale); otherwise it rounds to that scale and flags inexact,
        # offering more decimals via min_fixed_point_precision.
        ("sqrt(16)", None, "4 (exact)"),
        ("sqrt(2.25)", None, "1.50 (exact)"),  # 1.5 is exact at scale 2
        (
            "sqrt(2)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.4142)",
        ),
        (
            "sqrt(2.000000)",
            None,
            "1.414214 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.4142135624)",
        ),
        # binary64 sqrt is unconditionally inexact, even for a perfect square.
        ("sqrt(16)", "floating-point", "4.0 (inexact)"),
        ("sqrt(2)", "floating-point", "1.4142135623730951 (inexact)"),
        # Rational is exact only for a perfect square (both parts square).
        ("sqrt(16)", "rational", "4 (exact)"),
        ("sqrt(2.25)", "rational", "3/2 (exact)"),  # 9/4 -> 3/2
    ],
)
def test_sqrt(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # No real root for a negative operand in any mode...
        ("sqrt(-4)", None, "square root of a negative value"),
        ("sqrt(-4)", "floating-point", "square root of a negative value"),
        ("sqrt(-1)", "rational", "square root of a negative value"),
        # ...and rational refuses an irrational root (no scale to round to).
        ("sqrt(2)", "rational", "rational square root is irrational"),
    ],
)
def test_sqrt_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Fixed-point: exact only on a perfect cube at the operand's scale; otherwise
        # rounds to that scale and flags inexact.
        ("cbrt(8)", None, "2 (exact)"),
        ("cbrt(27)", None, "3 (exact)"),
        ("cbrt(0.008)", None, "0.200 (exact)"),  # 0.2 is a perfect cube at scale 3
        # Unlike sqrt, a negative is IN DOMAIN — an odd root carries the sign.
        ("cbrt(-8)", None, "-2 (exact)"),
        ("cbrt(-27.000)", None, "-3.000 (exact)"),
        (
            "cbrt(2)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.2599)",
        ),
        (
            "cbrt(2.000000)",
            None,
            "1.259921 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.2599210499)",
        ),
        # binary64 cbrt is unconditionally inexact, even for a perfect cube, and
        # handles negatives sign-preserving (no math.cbrt on the 3.10 floor).
        ("cbrt(8)", "floating-point", "2.0 (inexact)"),
        ("cbrt(-8)", "floating-point", "-2.0 (inexact)"),
        ("cbrt(2)", "floating-point", "1.2599210498948732 (inexact)"),
        # Rational is exact only when both parts are perfect cubes (negative numerator OK).
        ("cbrt(8)", "rational", "2 (exact)"),
        ("cbrt(27/8)", "rational", "3/2 (exact)"),
        ("cbrt(-8/27)", "rational", "-2/3 (exact)"),
    ],
)
def test_cbrt(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # No negative refusal (odd root): rational only refuses a non-perfect-cube.
        ("cbrt(2)", "rational", "rational cube root is irrational"),
        ("cbrt(1/2)", "rational", "rational cube root is irrational"),
        ("cbrt(-2)", "rational", "rational cube root is irrational"),
    ],
)
def test_cbrt_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # hypot (40.20): variadic Euclidean norm sqrt(x1**2 + ... + xn**2), inheriting
        # sqrt's stance. Fixed-point default — exact only when the sum of squares is a
        # perfect square at the covering scale; the squares accumulate exactly and round
        # ONCE at the root, so 1.5/2.0 land on 2.5 (a per-square fold would lose 2.25).
        ("hypot(3, 4)", None, "5 (exact)"),  # the 3-4-5 triangle
        ("hypot(5, 12)", None, "13 (exact)"),  # another Pythagorean triple
        ("hypot(20, 21)", None, "29 (exact)"),  # a non-multiple-of-5 triple
        ("hypot(1.5, 2.0)", None, "2.5 (exact)"),  # single rounding keeps it exact
        ("hypot(0.3, 0.4)", None, "0.5 (exact)"),
        # MIXED operand scales: the covering scale is max(decimals), and each mantissa is
        # rescaled by 10**(d - di) before summing — 0.30 (scale 2) with 0.4 (scale 1).
        ("hypot(0.30, 0.4)", None, "0.50 (exact)"),
        ("hypot(2, 3, 6)", None, "7 (exact)"),  # 4 + 9 + 36 = 49; VARIADIC, 3 coords
        ("hypot(6, 8, 0)", None, "10 (exact)"),  # a zero coordinate drops out (6-8-10)
        ("hypot(5)", None, "5 (exact)"),  # the lone-coordinate identity, |x|
        ("hypot(-5)", None, "5 (exact)"),  # every real in domain — squaring erases sign
        ("hypot(0)", None, "0 (exact)"),  # the origin
        (
            "hypot(1, 1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.4142)",
        ),
        (
            "hypot(1.000000, 1.000000)",
            None,
            "1.414214 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.4142135624)",
        ),
        # binary64 hypot is unconditionally inexact, even for a perfect norm; it uses the
        # type's own math.hypot, which stays finite where naive squaring would overflow.
        ("hypot(3, 4)", "floating-point", "5.0 (inexact)"),
        ("hypot(2, 3, 6)", "floating-point", "7.0 (inexact)"),
        ("hypot(5)", "floating-point", "5.0 (inexact)"),  # lone-coordinate identity, per mode
        ("hypot(3e200, 4e200)", "floating-point", "4.9999999999999995e+200 (inexact)"),
        # Rational is exact only when the sum of squares is a perfect square (both parts).
        ("hypot(3, 4)", "rational", "5 (exact)"),
        ("hypot(2, 3, 6)", "rational", "7 (exact)"),
        ("hypot(5)", "rational", "5 (exact)"),  # lone-coordinate identity stays exact here
        ("hypot(3/5, 4/5)", "rational", "1 (exact)"),  # 9/25 + 16/25 = 1
    ],
)
def test_hypot(expression, mode, value):
    assert _value(expression, mode) == value


def test_hypot_inexact_operand_poisons_the_result():
    # The exact flag is `all(operand.exact) and a perfect square`. A perfect norm built
    # from an INEXACT coordinate stays inexact — sqrt(2) is irrational, so hypot(3, sqrt(2))
    # has an inexact operand even though it does not itself land off the grid.
    assert _value("hypot(3, sqrt(2))").startswith("3 (inexact")


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # Rational refuses an irrational norm (no scale to round to) — the sum of
        # squares is not a perfect square.
        ("hypot(1, 1)", "rational", "rational euclidean norm is irrational"),
        ("hypot(1, 2)", "rational", "rational euclidean norm is irrational"),
    ],
)
def test_hypot_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Sum of squares (40.26): Σ xi**2. All-integer operands stay on the grid, so
        # exact in every mode — the fixed-point default here.
        ("sumsq(3, 4)", None, "25 (exact)"),  # 9 + 16
        ("sumsq(2, 3, 6)", None, "49 (exact)"),  # VARIADIC, 3 operands: 4 + 9 + 36
        ("sumsq(5)", None, "25 (exact)"),  # the lone-operand form is just x**2
        ("sumsq(-5)", None, "25 (exact)"),  # every real in domain — squaring erases sign
        ("sumsq(0)", None, "0 (exact)"),
        ("sumsq([1, 2, 3])", None, "14 (exact)"),  # a SINGLE vector reduces its ELEMENTS
        # The AVG stance (not hypot's): each square quantizes to the covering scale BEFORE
        # summing, so a sub-grid square rounds. 0.1**2 = 0.01 rounds to 0.0 at scale 1.
        (
            "sumsq(0.1)",
            None,
            "0.0 (inexact, rounded to 1 decimal — pass min_fixed_point_precision "
            "for more; e.g. =5 → 0.01000)",
        ),
        # 0.3**2=0.09→0.1 and 0.4**2=0.16→0.2 each round, then add exactly to 0.3 (true 0.25).
        (
            "sumsq(0.3, 0.4)",
            None,
            "0.3 (inexact, rounded to 1 decimal — pass min_fixed_point_precision "
            "for more; e.g. =5 → 0.25000)",
        ),
        # binary64 is unconditionally inexact, even for an integer sum of squares.
        ("sumsq(3, 4)", "floating-point", "25.0 (inexact)"),
        ("sumsq(2, 3, 6)", "floating-point", "49.0 (inexact)"),
        # Rational squares and adds exactly — no grid to fall off.
        ("sumsq(3, 4)", "rational", "25 (exact)"),
        ("sumsq(1/2, 1/2)", "rational", "1/2 (exact)"),  # 1/4 + 1/4
        ("sumsq(3/5, 4/5)", "rational", "1 (exact)"),  # 9/25 + 16/25
        ("sumsq([1, 2, 3])", "rational", "14 (exact)"),
    ],
)
def test_sumsq(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        # A vector is legal only as the SOLE operand — mixing it or emptying it refuses,
        # spelling out the two forms (the _series_operands contract shared with avg/variance).
        (
            "sumsq([1, 2], 3)",
            "sumsq has two forms — sumsq(vector) or sumsq(a, b, …) — and cannot mix them",
        ),
        ("sumsq([])", "sumsq of an empty vector is undefined"),
    ],
)
def test_sumsq_refuses_with_a_line_tagged_error(expression, error):
    payload = _calc(expression)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Geometric mean (40.15): the n-th root of the product. When the root lands on
        # the grid it is EXACT even in fixed-point — geomean(4, 9) = sqrt(36) = 6.
        ("geomean(4, 9)", None, "6 (exact)"),  # fixed-point default; a perfect square
        ("geomean(2, 8)", None, "4 (exact)"),  # sqrt(16)
        ("geomean(2, 4, 8)", None, "4 (exact)"),  # cube root of 64
        ("geomean(5)", None, "5 (exact)"),  # the lone-operand form is x itself
        ("geomean(0, 5)", None, "0 (exact)"),  # a zero factor zeroes the product
        ("geomean(2.00, 8.00)", None, "4.00 (exact)"),  # scale preserved
        ("geomean([4, 9])", None, "6 (exact)"),  # a SINGLE vector reduces its ELEMENTS
        # An irrational root rounds in fixed-point (widen the scale for accuracy) and in
        # float; here 24**(1/4) at the default scale 0 floors to 2.
        (
            "geomean(1, 2, 3, 4)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.2134)",
        ),
        # binary64 is unconditionally inexact, even on a perfect square.
        ("geomean(4, 9)", "floating-point", "6.0 (inexact)"),
        ("geomean(1, 2, 3, 4)", "floating-point", "2.213363839400643 (inexact)"),
        # Rational is exact ONLY for a perfect n-th power (both parts perfect powers).
        ("geomean(4, 9)", "rational", "6 (exact)"),
        ("geomean(8, 27, 64)", "rational", "24 (exact)"),  # cube root of 8*27*64 = 13824 = 24^3
        ("geomean(1/4, 1/9)", "rational", "1/6 (exact)"),  # sqrt of 1/36
        ("geomean([4, 9])", "rational", "6 (exact)"),
    ],
)
def test_geomean(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # DOMAIN non-negative: a negative operand makes an even root complex, refused
        # like sqrt's negative — checked per operand, so two negatives cannot cancel.
        ("geomean(-4, 9)", None, "geometric mean of a negative value"),
        ("geomean(-4, -9)", None, "geometric mean of a negative value"),
        ("geomean(-4, 9)", "floating-point", "geometric mean of a negative value"),
        # Rational refuses an irrational root rather than fabricate digits (exact-or-refuse).
        ("geomean(1, 2, 3, 4)", "rational", "rational nth root is irrational"),
        # The vector overload shares _series_operands' two-form contract.
        (
            "geomean([1, 2], 3)",
            None,
            "geomean has two forms — geomean(vector) or geomean(a, b, …) — and cannot mix them",
        ),
        ("geomean([])", None, "geomean of an empty vector is undefined"),
    ],
)
def test_geomean_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # Harmonic mean (40.16): n / Σ(1/xi). It composes the mode's own / and +, so like
        # avg it is EXACT in rational and rounds in fixed-point/float. Rational shows the
        # true value: harmean(1, 2, 4) = 3 / (1 + 1/2 + 1/4) = 3 / (7/4) = 12/7.
        ("harmean(1, 2, 4)", "rational", "12/7 (exact)"),
        ("harmean([2, 3])", "rational", "12/5 (exact)"),  # a SINGLE vector; 2/(1/2+1/3)
        ("harmean(6)", "rational", "6 (exact)"),  # the lone-operand form is x itself
        ("harmean(2, 3, 6)", "rational", "3 (exact)"),  # 3/(1/2+1/3+1/6) = 3/1
        # Fixed-point rounds each reciprocal to the covering scale — meaningful only at a
        # wider scale (the default scale 0 makes 1/2, 1/4 vanish), so widen precision.
        ("harmean(1, 2, 4)", "floating-point", "1.7142857142857142 (inexact)"),
        (
            "harmean(1, 2, 4)",
            4,  # min_fixed_point_precision via the floor argument
            "1.7143 (inexact, rounded to 4 decimals)",
        ),
    ],
)
def test_harmean(expression, mode, value):
    # `mode` doubles as the fixed-point floor when it is an int (the wide-scale case).
    if isinstance(mode, int):
        assert _value(expression, None, mode) == value
    else:
        assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # DOMAIN positive: a NEGATIVE operand (mixed signs are ill-defined) refuses here,
        # while a ZERO makes 1/xi undefined and divides by zero — the two failure modes.
        ("harmean(-1, 2)", "rational", "harmonic mean of a non-positive value"),
        ("harmean(2, -3)", "rational", "harmonic mean of a non-positive value"),
        ("harmean(0, 4)", None, "fixed-point division by zero"),  # zero factor: 1/0
        # The vector overload shares _series_operands' two-form contract.
        (
            "harmean([1, 2], 3)",
            None,
            "harmean has two forms — harmean(vector) or harmean(a, b, …) — and cannot mix them",
        ),
        ("harmean([])", None, "harmean of an empty vector is undefined"),
    ],
)
def test_harmean_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    "name",
    ["hypot", "sumsq", "geomean", "harmean", "avg", "max", "min", "variance", "gcd", "lcm"],
)
def test_variadic_functions_refuse_below_their_arity_floor(name):
    # A variadic (1, None) function needs at least one operand; calling it with none is a
    # parse-time arity error, not a domain refusal. Pins the lower-bound wording for the
    # whole family (the (1, None) arity declares a MINIMUM, not a fixed count).
    payload = _calc(f"{name}()")
    assert payload["error"] == f"function '{name}' takes at least 1 argument(s), but 0 given"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # factorial (40.4): n! for a non-negative integer, EXACT in every mode — a
        # product of integers with no rounding.
        ("factorial(0)", None, "1 (exact)"),  # fixed-point default; 0! = 1
        ("factorial(5)", None, "120 (exact)"),
        ("factorial(10)", None, "3628800 (exact)"),
        ("factorial(5.00)", None, "120 (exact)"),  # whole-valued fixed-point literal is in domain
        ("factorial(0)", "rational", "1 (exact)"),
        ("factorial(20)", "rational", "2432902008176640000 (exact)"),  # exact bignum, no scale
        # binary64: exact while the double represents n! precisely (every n <= 18; 20!
        # still lands exactly via its trailing factors of two), inexact once it cannot.
        ("factorial(5)", "floating-point", "120.0 (exact)"),
        ("factorial(18)", "floating-point", "6402373705728000.0 (exact)"),
        ("factorial(20)", "floating-point", "2.43290200817664e+18 (exact)"),
        ("factorial(25)", "floating-point", "1.5511210043330986e+25 (inexact)"),
    ],
)
def test_factorial(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # A negative operand is the gamma extension (40.4.1), refused here in every mode...
        ("factorial(-1)", None, "factorial of a negative value"),
        ("factorial(-1)", "floating-point", "factorial of a negative value"),
        ("factorial(-1)", "rational", "factorial of a negative value"),
        # ...as is a non-integer operand (the integer-domain gate, shared with gcd/lcm).
        ("factorial(2.5)", None, "factorial requires integer operands"),
        ("factorial(2.5)", "floating-point", "factorial requires integer operands"),
        ("factorial(1/2)", "rational", "factorial requires integer operands"),
        # n is capped so a huge operand cannot blow up...
        ("factorial(1001)", None, "factorial argument too large (limit 1000)"),
        ("factorial(1001)", "rational", "factorial argument too large (limit 1000)"),
        # ...and float refuses far below the cap, where n! overflows a double (~n>170).
        ("factorial(171)", "floating-point", "factorial overflows floating-point"),
    ],
)
def test_factorial_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # comb (40.5): binomial coefficient C(n, k), the count of k-subsets of n,
        # EXACT in every mode — the multiplicative form cancels to an integer.
        ("comb(5, 2)", None, "10 (exact)"),  # fixed-point default
        ("comb(0, 0)", None, "1 (exact)"),  # empty set, the one subset
        ("comb(10, 0)", None, "1 (exact)"),  # C(n, 0) = 1
        ("comb(10, 10)", None, "1 (exact)"),  # C(n, n) = 1
        ("comb(52, 5)", None, "2598960 (exact)"),  # five-card poker hands
        ("comb(5.00, 2.00)", None, "10 (exact)"),  # whole-valued fixed-point literals are in domain
        # out-of-range k chooses an impossible subset and is 0...
        ("comb(5, -1)", None, "0 (exact)"),  # k < 0
        ("comb(3, 5)", None, "0 (exact)"),  # k > n
        ("comb(-1, 0)", None, "0 (exact)"),  # negative n: every k >= 0 exceeds n
        # rational: the exact integer at scale 0, like factorial.
        ("comb(6, 2)", "rational", "15 (exact)"),
        ("comb(40, 20)", "rational", "137846528820 (exact)"),
        # binary64: exact while the double represents C(n, k) precisely, inexact once it cannot.
        ("comb(5, 2)", "floating-point", "10.0 (exact)"),
        ("comb(52, 5)", "floating-point", "2598960.0 (exact)"),
        ("comb(100, 50)", "floating-point", "1.008913445455642e+29 (inexact)"),
    ],
)
def test_comb(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # A non-integer operand is the gamma-generalized coefficient (40.5.1), refused
        # here in every mode by the integer-domain gate shared with factorial/gcd.
        ("comb(2.5, 2)", None, "comb requires integer operands"),
        ("comb(5, 2.5)", "floating-point", "comb requires integer operands"),
        ("comb(5, 1/2)", "rational", "comb requires integer operands"),
        # The term count min(k, n-k) is capped so a huge operand cannot blow up.
        ("comb(5000, 2500)", None, "comb argument too large (limit 1000)"),
        ("comb(5000, 2500)", "rational", "comb argument too large (limit 1000)"),
    ],
)
def test_comb_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # perm (40.6): falling factorial P(n, k) = n!/(n-k)!, ordered k-permutations
        # of n, EXACT in every mode — a product of k consecutive integers.
        ("perm(5, 2)", None, "20 (exact)"),  # fixed-point default
        ("perm(5, 0)", None, "1 (exact)"),  # P(n, 0) = 1, the empty arrangement
        ("perm(5, 5)", None, "120 (exact)"),  # P(n, n) = n!
        ("perm(10, 3)", None, "720 (exact)"),
        ("perm(52, 5)", None, "311875200 (exact)"),  # ordered five-card deals
        ("perm(5.00, 2.00)", None, "20 (exact)"),  # whole-valued fixed-point literals are in domain
        # out-of-range k arranges an impossible selection and is 0...
        ("perm(5, -1)", None, "0 (exact)"),  # k < 0
        ("perm(3, 5)", None, "0 (exact)"),  # k > n
        ("perm(-1, 0)", None, "0 (exact)"),  # negative n: every k >= 0 exceeds n
        # rational: the exact integer at scale 0, like comb.
        ("perm(6, 2)", "rational", "30 (exact)"),
        ("perm(20, 10)", "rational", "670442572800 (exact)"),
        # binary64: P(n, n) = n! so it tracks factorial — exact while the double
        # represents the integer precisely, inexact once it cannot.
        ("perm(5, 2)", "floating-point", "20.0 (exact)"),
        ("perm(18, 18)", "floating-point", "6402373705728000.0 (exact)"),
        ("perm(25, 25)", "floating-point", "1.5511210043330986e+25 (inexact)"),
    ],
)
def test_perm(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # A non-integer operand is the gamma-generalized permutation (40.6.1), refused
        # here in every mode by the integer-domain gate shared with comb/factorial.
        ("perm(2.5, 2)", None, "perm requires integer operands"),
        ("perm(5, 2.5)", "floating-point", "perm requires integer operands"),
        ("perm(5, 1/2)", "rational", "perm requires integer operands"),
        # The term count k is capped so a huge operand cannot blow up...
        ("perm(2000, 1001)", None, "perm argument too large (limit 1000)"),
        ("perm(2000, 1001)", "rational", "perm argument too large (limit 1000)"),
        # ...and float refuses where P(n, n) = n! overflows a double (~n>170).
        ("perm(171, 171)", "floating-point", "perm overflows floating-point"),
    ],
)
def test_perm_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # clamp (40.21): constrain x to [lo, hi] = min(hi, max(lo, x)). SELECTION not
        # math — returns one of the three operands verbatim, so EXACT in every mode and
        # carrying the chosen operand's own scale.
        ("clamp(5, 0, 10)", None, "5 (exact)"),  # within range -> x
        ("clamp(-3, 0, 10)", None, "0 (exact)"),  # below -> lo
        ("clamp(15, 0, 10)", None, "10 (exact)"),  # above -> hi
        ("clamp(0, 0, 10)", None, "0 (exact)"),  # at lo boundary
        ("clamp(10, 0, 10)", None, "10 (exact)"),  # at hi boundary
        ("clamp(2.50, 0, 10)", None, "2.50 (exact)"),  # within -> x kept at its own scale
        ("clamp(15, 0, 2.5)", None, "2.5 (exact)"),  # above -> hi kept at its own scale
        ("clamp(1.5, 1.50, 3)", None, "1.5 (exact)"),  # bounds compare across scales (1.5 == 1.50)
        # rational: exact selection, no arithmetic.
        ("clamp(5, 0, 10)", "rational", "5 (exact)"),
        ("clamp(7/2, 1, 3)", "rational", "3 (exact)"),  # 3.5 clamped down to hi
        # floating-point: still pure selection; the chosen operand carries binary64's flag.
        ("clamp(0.1 + 0.2, 0, 1)", "floating-point", "0.30000000000000004 (inexact)"),
    ],
)
def test_clamp(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # DOMAIN lo <= hi: an inverted range is meaningless and refuses in every mode.
        ("clamp(5, 10, 0)", None, "clamp requires lo <= hi"),
        ("clamp(5, 10, 0)", "rational", "clamp requires lo <= hi"),
        ("clamp(5, 10, 0)", "floating-point", "clamp requires lo <= hi"),
    ],
)
def test_clamp_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # lerp (40.22): linear interpolation a + (b - a)*t. Plain ARITHMETIC, so it
        # follows the avg/division stance — endpoints land exactly, the interior may
        # round in fixed-point/float.
        ("lerp(0, 10, 0.5)", None, "5.0 (exact)"),  # midpoint
        ("lerp(0, 10, 0)", None, "0 (exact)"),  # t=0 returns a
        ("lerp(0, 10, 1)", None, "10 (exact)"),  # t=1 returns b
        ("lerp(2, 8, 0.25)", None, "3.50 (exact)"),
        ("lerp(10, 0, 0.5)", None, "5.0 (exact)"),  # b below a interpolates down
        ("lerp(0, 10, 2)", None, "20 (exact)"),  # t outside [0, 1] extrapolates
        # fixed-point MAY round where the *t multiply leaves the grid (1/3 at scale 0).
        (
            "lerp(0, 1, 1/3)",
            None,
            "0 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.3333)",
        ),
        # rational is EXACT — the same interior point keeps its fraction.
        ("lerp(0, 1, 1/3)", "rational", "1/3 (exact)"),
        ("lerp(0, 10, 7/3)", "rational", "70/3 (exact)"),  # exact extrapolation
        # floating-point always carries binary64's inexact flag.
        ("lerp(0, 10, 0.5)", "floating-point", "5.0 (inexact)"),
    ],
)
def test_lerp(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "floor", "value"),
    [
        # With a precision floor the fixed-point interior shows its rounding: the true
        # midpoint-third is 4, but 1/3 rounds, so 2 + 6*0.3333 lands at 3.9998 (40.22).
        ("lerp(2, 8, 1/3)", 4, "3.9998 (inexact, rounded to 4 decimals)"),
    ],
)
def test_lerp_fixed_point_rounds_with_a_precision_floor(expression, floor, value):
    assert _value(expression, None, floor) == value


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # floor (28.23): round toward -inf with an optional ndigits count (default 0).
        # Fixed-point snaps to the grid, always exact.
        ("floor(2.7)", None, "2 (exact)"),
        ("floor(-2.1)", None, "-3 (exact)"),  # toward -inf, not toward zero
        ("floor(5)", None, "5 (exact)"),
        ("floor(2.756, 2)", None, "2.75 (exact)"),  # floor at scale 2
        ("floor(2.756, 5)", None, "2.75600 (exact)"),  # finer than the value: held at scale 5
        ("floor(1234, -2)", None, "1200 (exact)"),  # negative ndigits -> tens/hundreds
        # Floating-point: exact when it lands on an integer (ndigits <= 0), inexact
        # when ndigits > 0 (the n-decimal target is not binary-representable).
        ("floor(2.7)", "floating-point", "2.0 (exact)"),
        ("floor(-2.1)", "floating-point", "-3.0 (exact)"),
        ("floor(1234, -2)", "floating-point", "1200.0 (exact)"),
        ("floor(2.756, 2)", "floating-point", "2.75 (inexact)"),
        # Rational: exact in every case (Fraction floor of the shifted value).
        ("floor(2.7)", "rational", "2 (exact)"),
        ("floor(-2.1)", "rational", "-3 (exact)"),
        ("floor(1234, -2)", "rational", "1200 (exact)"),
        ("floor(2.756, 2)", "rational", "11/4 (exact)"),  # 275/100 -> 11/4
        ("floor(1/3)", "rational", "0 (exact)"),
        ("floor(1/3, 2)", "rational", "33/100 (exact)"),
    ],
)
def test_floor(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # ndigits is a COUNT, required to be an integer in any mode — a fractional
        # second argument refuses (the pow integer-exponent stance, 28.20/28.22).
        ("floor(2.5, 1.5)", None, "ndigits must be an integer"),
        ("floor(2.5, 0.5)", "floating-point", "ndigits must be an integer"),
        ("floor(2.5, 3/2)", "rational", "ndigits must be an integer"),
    ],
)
def test_floor_refuses_a_non_integer_ndigits(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # ceil (28.24): round toward +inf, the mirror of floor. Fixed-point snaps to
        # the grid, always exact.
        ("ceil(2.1)", None, "3 (exact)"),
        ("ceil(-2.7)", None, "-2 (exact)"),  # toward +inf, not away from zero
        ("ceil(5)", None, "5 (exact)"),
        ("ceil(2.751, 2)", None, "2.76 (exact)"),  # ceil at scale 2
        ("ceil(2.75, 5)", None, "2.75000 (exact)"),  # finer than the value: held at scale 5
        ("ceil(1234, -2)", None, "1300 (exact)"),  # negative ndigits -> tens/hundreds
        # Floating-point: exact when it lands on an integer (ndigits <= 0), inexact
        # when ndigits > 0 (the n-decimal target is not binary-representable).
        ("ceil(2.1)", "floating-point", "3.0 (exact)"),
        ("ceil(-2.7)", "floating-point", "-2.0 (exact)"),
        ("ceil(1234, -2)", "floating-point", "1300.0 (exact)"),
        ("ceil(2.751, 2)", "floating-point", "2.76 (inexact)"),
        # Rational: exact in every case (Fraction ceil of the shifted value).
        ("ceil(2.1)", "rational", "3 (exact)"),
        ("ceil(-2.7)", "rational", "-2 (exact)"),
        ("ceil(1234, -2)", "rational", "1300 (exact)"),
        ("ceil(2.751, 2)", "rational", "69/25 (exact)"),  # 276/100 -> 69/25
        ("ceil(1/3)", "rational", "1 (exact)"),
        ("ceil(1/3, 2)", "rational", "17/50 (exact)"),  # 34/100 -> 17/50
    ],
)
def test_ceil(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # ndigits must be an integer in any mode — a fractional second argument
        # refuses, exactly as floor (28.22/28.24).
        ("ceil(2.5, 1.5)", None, "ndigits must be an integer"),
        ("ceil(2.5, 0.5)", "floating-point", "ndigits must be an integer"),
        ("ceil(2.5, 3/2)", "rational", "ndigits must be an integer"),
    ],
)
def test_ceil_refuses_a_non_integer_ndigits(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # round (28.25): nearest, ties to EVEN (banker's). The defining ties — half
        # goes to the even neighbour, NOT away from zero. Fixed-point is exact.
        ("round(2.5)", None, "2 (exact)"),  # tie -> even (2), not 3
        ("round(3.5)", None, "4 (exact)"),  # tie -> even (4)
        ("round(-2.5)", None, "-2 (exact)"),  # tie -> even (-2)
        ("round(0.5)", None, "0 (exact)"),
        ("round(2.1)", None, "2 (exact)"),  # nearest, no tie
        ("round(2.345, 2)", None, "2.34 (exact)"),  # true-decimal tie -> even (2.34)
        ("round(2.355, 2)", None, "2.36 (exact)"),  # tie -> even (2.36)
        ("round(2.75, 5)", None, "2.75000 (exact)"),  # finer than the value: held at scale 5
        ("round(1250, -2)", None, "1200 (exact)"),  # tie -> even hundred (1200)
        ("round(1350, -2)", None, "1400 (exact)"),  # tie -> even hundred (1400)
        # Floating-point: half-even via the builtin; exact when it lands on an
        # integer (ndigits <= 0), inexact when ndigits > 0.
        ("round(2.5)", "floating-point", "2.0 (exact)"),
        ("round(3.5)", "floating-point", "4.0 (exact)"),
        ("round(-2.5)", "floating-point", "-2.0 (exact)"),
        ("round(1250, -2)", "floating-point", "1200.0 (exact)"),
        # 2.345 is just above 2.345 in binary, so the double rounds UP to 2.35 —
        # diverging from fixed-point/rational's true-decimal 2.34, and inexact.
        ("round(2.345, 2)", "floating-point", "2.35 (inexact)"),
        # Rational: half-even via Fraction.__round__, exact in every case.
        ("round(2.5)", "rational", "2 (exact)"),
        ("round(3.5)", "rational", "4 (exact)"),
        ("round(-2.5)", "rational", "-2 (exact)"),
        ("round(1250, -2)", "rational", "1200 (exact)"),
        ("round(2.345, 2)", "rational", "117/50 (exact)"),  # 234/100 -> 117/50 (the true 2.34)
        ("round(1/3)", "rational", "0 (exact)"),
        ("round(1/3, 2)", "rational", "33/100 (exact)"),
    ],
)
def test_round(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # ndigits must be an integer in any mode — a fractional second argument
        # refuses, exactly as floor/ceil (28.22/28.25).
        ("round(2.5, 1.5)", None, "ndigits must be an integer"),
        ("round(2.5, 0.5)", "floating-point", "ndigits must be an integer"),
        ("round(2.5, 3/2)", "rational", "ndigits must be an integer"),
    ],
)
def test_round_refuses_a_non_integer_ndigits(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # trunc (28.26): round toward ZERO — drop the fraction. The defining contrast
        # with floor is on NEGATIVES (toward zero, not toward -inf). Fixed-point exact.
        ("trunc(2.7)", None, "2 (exact)"),
        ("trunc(-2.7)", None, "-2 (exact)"),  # toward zero -> -2 (floor gives -3)
        ("trunc(-2.1)", None, "-2 (exact)"),
        ("trunc(5)", None, "5 (exact)"),
        ("trunc(2.759, 2)", None, "2.75 (exact)"),  # trunc at scale 2
        ("trunc(-2.759, 2)", None, "-2.75 (exact)"),  # magnitude dropped, not -2.76
        ("trunc(2.75, 5)", None, "2.75000 (exact)"),  # finer than the value: held at scale 5
        ("trunc(1290, -2)", None, "1200 (exact)"),  # negative ndigits -> tens/hundreds
        ("trunc(-1290, -2)", None, "-1200 (exact)"),
        # Floating-point: exact when it lands on an integer (ndigits <= 0), inexact
        # when ndigits > 0 (the n-decimal target is not binary-representable).
        ("trunc(2.7)", "floating-point", "2.0 (exact)"),
        ("trunc(-2.7)", "floating-point", "-2.0 (exact)"),
        ("trunc(1290, -2)", "floating-point", "1200.0 (exact)"),
        ("trunc(2.759, 2)", "floating-point", "2.75 (inexact)"),
        # Rational: exact in every case (Fraction trunc of the shifted value).
        ("trunc(2.7)", "rational", "2 (exact)"),
        ("trunc(-2.7)", "rational", "-2 (exact)"),
        ("trunc(1290, -2)", "rational", "1200 (exact)"),
        ("trunc(2.759, 2)", "rational", "11/4 (exact)"),  # 275/100 -> 11/4
        ("trunc(-7/2)", "rational", "-3 (exact)"),  # -3.5 toward zero -> -3 (floor gives -4)
        ("trunc(1/3, 2)", "rational", "33/100 (exact)"),
    ],
)
def test_trunc(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # ndigits must be an integer in any mode — a fractional second argument
        # refuses, exactly as the rest of the family (28.22/28.26).
        ("trunc(2.5, 1.5)", None, "ndigits must be an integer"),
        ("trunc(2.5, 0.5)", "floating-point", "ndigits must be an integer"),
        ("trunc(2.5, 3/2)", "rational", "ndigits must be an integer"),
    ],
)
def test_trunc_refuses_a_non_integer_ndigits(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # The BINARY pow(x, y) function — the call form of ** (28.20). Integer
        # exponent is exact in the exact modes, float rounds.
        ("pow(2, 10)", None, "1024 (exact)"),
        ("pow(2, 10)", "rational", "1024 (exact)"),
        ("pow(2, 10)", "floating-point", "1024.0 (inexact)"),
        # FRACTIONAL fixed-point exponent (28.20.1). PATH A — a perfect q-th root is
        # exact (result scale covers base and exponent).
        ("pow(4, 0.5)", None, "2.0 (exact)"),  # sqrt(4)
        ("pow(0.25, 0.5)", None, "0.50 (exact)"),  # sqrt(1/4)
        ("pow(32, 0.2)", None, "2.0 (exact)"),  # fifth root of 32
        # PATH B — an irrational root computes via exp(y*ln x), inexact.
        (
            "pow(2.000000, 0.5)",
            None,
            "1.414214 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.4142135624)",
        ),
        ("pow(2, 0.5)", "floating-point", "1.4142135623730951 (inexact)"),
    ],
)
def test_pow(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # A non-integer exponent is irrational for a rational base (no scale).
        ("pow(2, 0.5)", "rational", "rational power requires an integer exponent"),
        # An even root of a negative base is complex — no real value.
        ("pow(-4, 0.5)", None, "even root of a negative value"),
        # An odd irrational root of a negative base cannot go through ln — refuse.
        ("pow(-2, 0.2)", None, "fractional power of a negative base is irrational"),
        # Zero to a negative power divides by zero.
        ("pow(0, -2)", None, "fixed-point zero to a negative power"),
    ],
)
def test_pow_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # sin(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("sin(0)", None, "0 (exact)"),
        ("sin(0)", "rational", "0 (exact)"),
        ("sin(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.sin and is unconditionally inexact.
        ("sin(1)", "floating-point", "0.8414709848078965 (inexact)"),
        ("sin(0.5)", "floating-point", "0.479425538604203 (inexact)"),
        # Fixed-point sums a Taylor series at the operand's scale and flags inexact;
        # the result rounds to that scale (sin(1) ~ 0.8415 -> 1 at scale 0).
        (
            "sin(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.8415)",
        ),
        # Inexact fixed-point results also carry the min_fixed_point_precision offer
        # hint (the worked example at scale+4), exactly like sqrt(2.000000).
        (
            "sin(1.000000)",  # math.sin(1) at scale 6
            None,
            "0.841471 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.8414709848)",
        ),
        (
            "sin(0.500000)",
            None,
            "0.479426 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.4794255386)",
        ),
        # Negative argument (sin is odd) — exercises the sign-fold.
        (
            "sin(-1.500000)",
            None,
            "-0.997495 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.9974949866)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "sin(10.000000)",  # math.sin(10)
            None,
            "-0.544021 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.5440211109)",
        ),
        (
            "sin(1000.000000)",  # math.sin(1000), heavily range-reduced
            None,
            "0.826880 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.8268795405)",
        ),
    ],
)
def test_sin(expression, mode, value):
    assert _value(expression, mode) == value


def test_sin_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("sin(1)", "rational")
    assert payload["error"] == "sine of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # cos(0) = 1 is the one exact case in every mode (transcendental otherwise).
        ("cos(0)", None, "1 (exact)"),
        ("cos(0)", "rational", "1 (exact)"),
        ("cos(0)", "floating-point", "1.0 (inexact)"),
        # binary64 uses math.cos and is unconditionally inexact.
        ("cos(1)", "floating-point", "0.5403023058681398 (inexact)"),
        ("cos(0.5)", "floating-point", "0.8775825618903728 (inexact)"),
        # Fixed-point sums the even Taylor series at the operand's scale, inexact;
        # cos(1) ~ 0.5403 rounds to 1 at scale 0 (nearest whole number).
        (
            "cos(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.5403)",
        ),
        (
            "cos(1.000000)",  # math.cos(1) at scale 6
            None,
            "0.540302 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.5403023059)",
        ),
        (
            "cos(0.500000)",
            None,
            "0.877583 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.8775825619)",
        ),
        # cos is even — a negative argument gives the same result as its magnitude.
        (
            "cos(-2.000000)",
            None,
            "-0.416147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.4161468365)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "cos(10.000000)",  # math.cos(10)
            None,
            "-0.839072 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.8390715291)",
        ),
    ],
)
def test_cos(expression, mode, value):
    assert _value(expression, mode) == value


def test_cos_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("cos(1)", "rational")
    assert payload["error"] == "cosine of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # tan(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("tan(0)", None, "0 (exact)"),
        ("tan(0)", "rational", "0 (exact)"),
        ("tan(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.tan and is unconditionally inexact.
        ("tan(1)", "floating-point", "1.5574077246549023 (inexact)"),
        ("tan(0.5)", "floating-point", "0.5463024898437905 (inexact)"),
        # Fixed-point divides the sine and cosine series at the working scale, inexact;
        # tan(1) ~ 1.5574 rounds to 2 at scale 0 (nearest whole number).
        (
            "tan(1)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5574)",
        ),
        (
            "tan(1.000000)",  # math.tan(1) at scale 6
            None,
            "1.557408 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5574077247)",
        ),
        (
            "tan(0.500000)",
            None,
            "0.546302 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.5463024898)",
        ),
        # Negative argument (tan is odd) — exercises the sine sign-fold.
        (
            "tan(-1.000000)",
            None,
            "-1.557408 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -1.5574077247)",
        ),
        # Second quadrant: cos is negative, so tan comes out negative.
        (
            "tan(2.000000)",  # math.tan(2)
            None,
            "-2.185040 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -2.1850398633)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "tan(10.000000)",  # math.tan(10)
            None,
            "0.648361 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6483608275)",
        ),
        # Near an odd multiple of pi/2 the cosine is tiny but never exactly 0 (pi is
        # irrational), so the answer is just a large inexact value, not an error.
        (
            "tan(1.570796)",  # math.tan(1.570796), just shy of pi/2
            None,
            "3060023.306194 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 3060023.3061935647)",
        ),
    ],
)
def test_tan(expression, mode, value):
    assert _value(expression, mode) == value


def test_tan_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("tan(1)", "rational")
    assert payload["error"] == "tangent of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # cot == cos/sin, the mirror of tan: transcendental, so inexact everywhere it
        # is defined — and UNLIKE tan there is no exact case, since cot(0) is undefined
        # (sin = 0). binary64 uses math.cos / math.sin and is unconditionally inexact.
        ("cot(1)", "floating-point", "0.6420926159343308 (inexact)"),
        ("cot(0.5)", "floating-point", "1.830487721712452 (inexact)"),
        # Fixed-point divides the cosine and sine series at the working scale, inexact;
        # cot(1) ~ 0.6421 rounds to 1 at scale 0 (nearest whole number).
        (
            "cot(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.6421)",
        ),
        (
            "cot(1.000000)",  # cos/sin of 1 at scale 6
            None,
            "0.642093 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6420926159)",
        ),
        (
            "cot(0.500000)",
            None,
            "1.830488 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.8304877217)",
        ),
        # Negative argument (cot is odd) — exercises the sine sign-fold.
        (
            "cot(-1.000000)",
            None,
            "-0.642093 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.6420926159)",
        ),
        # Second quadrant: cos is negative, so cot comes out negative.
        (
            "cot(2.000000)",  # cos/sin of 2
            None,
            "-0.457658 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.4576575544)",
        ),
        # Range reduction mod 2*pi for arguments well outside (-pi, pi].
        (
            "cot(10.000000)",  # cos/sin of 10
            None,
            "1.542351 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5423510454)",
        ),
        # Near a multiple of pi the sine is tiny but never exactly 0 (pi is
        # irrational), so the answer is just a large inexact value, not an error.
        (
            "cot(3.141592)",  # cos/sin of 3.141592, just shy of pi
            None,
            "-1530011.653097 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -1530011.6530966189)",
        ),
    ],
)
def test_cot(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        # cot(0) = cos/sin = 1/0 is undefined in every mode (the mirror of tan(0) = 0,
        # which IS defined): sin = 0 there, so each mode raises its division/domain error.
        (None, "fixed-point cotangent of a multiple of pi"),
        ("rational", "cotangent of zero is undefined"),
        ("floating-point", "float division by zero"),
    ],
)
def test_cot_of_zero_is_undefined_with_a_line_tagged_error(mode, error):
    payload = _calc("cot(0)", mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_cot_refuses_a_non_zero_rational_with_a_line_tagged_error():
    payload = _calc("cot(1)", "rational")
    assert payload["error"] == "cotangent of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # asin(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("asin(0)", None, "0 (exact)"),
        ("asin(0)", "rational", "0 (exact)"),
        ("asin(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.asin and is unconditionally inexact; asin(1) = pi/2.
        ("asin(1)", "floating-point", "1.5707963267948966 (inexact)"),
        ("asin(0.5)", "floating-point", "0.5235987755982989 (inexact)"),
        # asin is odd — a negative argument negates the result (the sign-fold).
        ("asin(-0.5)", "floating-point", "-0.5235987755982989 (inexact)"),
        # Fixed-point routes through the arctan series and flags inexact; asin(1) =
        # pi/2 ~ 1.5708 rounds to 2 at scale 0 (nearest whole number).
        (
            "asin(1)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5708)",
        ),
        # asin(1/2) = pi/6; the inexact result carries the precision-offer hint.
        (
            "asin(0.500000)",
            None,
            "0.523599 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.5235987756)",
        ),
        # Negative argument (asin is odd) — exercises the fixed-point sign-fold.
        (
            "asin(-0.500000)",
            None,
            "-0.523599 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.5235987756)",
        ),
        # |x| > 1/sqrt(2): the arctan argument exceeds 1, so asin reduces via
        # pi/2 - atan(1/u). asin(sqrt(3)/2) ~ pi/3 (0.866025 is just shy of it).
        (
            "asin(0.866025)",
            None,
            "1.047197 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.0471967436)",
        ),
        # The domain endpoint asin(1) = pi/2 at a non-zero scale (root = 0 path).
        (
            "asin(1.000000)",
            None,
            "1.570796 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5707963268)",
        ),
    ],
)
def test_asin(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    "mode",
    [None, "floating-point", "rational"],
)
def test_asin_refuses_outside_the_domain_with_a_line_tagged_error(mode):
    # |x| > 1 has no real arcsine: every mode refuses with the same domain message.
    payload = _calc("asin(2)", mode)
    assert payload["error"] == "arcsine argument outside the domain [-1, 1]"
    assert payload["value"] is None


def test_asin_refuses_a_non_zero_rational_with_a_line_tagged_error():
    # In-domain but irrational: a non-zero rational arcsine refuses (exact-or-refuse).
    payload = _calc("asin(0.5)", "rational")
    assert payload["error"] == "arcsine of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # acos(1) = 0 is the one exact case in every mode (acos = pi/2 - asin, and
        # asin(1) = pi/2 cancels). NB the exact landmark is x = 1, not x = 0.
        ("acos(1)", None, "0 (exact)"),
        ("acos(1)", "rational", "0 (exact)"),
        ("acos(1)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.acos and is unconditionally inexact; acos(0) = pi/2.
        ("acos(0)", "floating-point", "1.5707963267948966 (inexact)"),
        ("acos(0.5)", "floating-point", "1.0471975511965979 (inexact)"),
        # acos is NOT odd: acos(-x) = pi - acos(x), so a negative argument lands in
        # (pi/2, pi]. acos(-1) = pi is the far endpoint.
        ("acos(-0.5)", "floating-point", "2.0943951023931957 (inexact)"),
        ("acos(-1)", "floating-point", "3.141592653589793 (inexact)"),
        # Fixed-point subtracts asin from pi/2 and flags inexact; acos(0) = pi/2 ~
        # 1.5708 rounds to 2 at scale 0 (acos(0) is irrational, unlike asin(0) = 0).
        (
            "acos(0)",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 1.5708)",
        ),
        # acos(1/2) = pi/3; the inexact result carries the precision-offer hint.
        (
            "acos(0.500000)",
            None,
            "1.047198 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.0471975512)",
        ),
        # Negative argument (acos(-1/2) = 2*pi/3) — exercises the sign handling.
        (
            "acos(-0.500000)",
            None,
            "2.094395 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 2.0943951024)",
        ),
        # acos(-1) = pi, the far domain endpoint.
        (
            "acos(-1.000000)",
            None,
            "3.141593 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 3.1415926536)",
        ),
        # acos(1) = 0 stays EXACT at a non-zero scale (the mantissa == 10**decimals path).
        ("acos(1.000000)", None, "0.000000 (exact)"),
    ],
)
def test_acos(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    "mode",
    [None, "floating-point", "rational"],
)
def test_acos_refuses_outside_the_domain_with_a_line_tagged_error(mode):
    # |x| > 1 has no real arccosine: every mode refuses with the same domain message.
    payload = _calc("acos(2)", mode)
    assert payload["error"] == "arccosine argument outside the domain [-1, 1]"
    assert payload["value"] is None


def test_acos_refuses_a_rational_other_than_one_with_a_line_tagged_error():
    # In-domain but irrational: any rational != 1 arccosine refuses (exact-or-refuse).
    payload = _calc("acos(0.5)", "rational")
    assert payload["error"] == "arccosine of a rational other than 1 is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # atan(0) = 0 is the one exact case in every mode (transcendental otherwise).
        ("atan(0)", None, "0 (exact)"),
        ("atan(0)", "rational", "0 (exact)"),
        ("atan(0)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.atan and is unconditionally inexact.
        ("atan(0.5)", "floating-point", "0.4636476090008061 (inexact)"),
        ("atan(1)", "floating-point", "0.7853981633974483 (inexact)"),  # pi/4
        ("atan(2)", "floating-point", "1.1071487177940904 (inexact)"),  # over-1 argument
        # atan is odd: atan(-x) = -atan(x).
        ("atan(-0.5)", "floating-point", "-0.4636476090008061 (inexact)"),
        ("atan(-2)", "floating-point", "-1.1071487177940904 (inexact)"),
        # No domain limit — a large argument just approaches pi/2 from below.
        ("atan(1000)", "floating-point", "1.5697963271282298 (inexact)"),
        # Fixed-point sums the arctan series and flags inexact; atan(1) = pi/4 ~ 0.7854
        # rounds to 1 at scale 0 (atan(1) is irrational, the only exact case is x = 0).
        (
            "atan(1)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.7854)",
        ),
        # atan(1/2); the inexact result carries the precision-offer hint.
        (
            "atan(0.500000)",
            None,
            "0.463648 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.4636476090)",
        ),
        # An over-1 argument exercises the atan(x) = pi/2 - atan(1/x) reduction.
        (
            "atan(2.000000)",
            None,
            "1.107149 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.1071487178)",
        ),
        # Negative argument (atan is odd) — exercises the sign handling.
        (
            "atan(-0.500000)",
            None,
            "-0.463648 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.4636476090)",
        ),
        (
            "atan(-2.000000)",
            None,
            "-1.107149 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -1.1071487178)",
        ),
        # A large argument nears pi/2 ~ 1.5708 without ever reaching it (no domain cap).
        (
            "atan(1000.000000)",
            None,
            "1.569796 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5697963271)",
        ),
        # atan(0) = 0 stays EXACT at a non-zero scale (the mantissa == 0 path).
        ("atan(0.000000)", None, "0.000000 (exact)"),
    ],
)
def test_atan(expression, mode, value):
    assert _value(expression, mode) == value


def test_atan_refuses_a_non_zero_rational_with_a_line_tagged_error():
    # atan of a rational is irrational except atan(0) = 0; any non-zero rational refuses
    # (exact-or-refuse). NB no domain refusal — atan accepts every real.
    payload = _calc("atan(2)", "rational")
    assert payload["error"] == "arctangent of a non-zero rational is irrational"
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # degrees(x) = x*180/pi, radians(x) = x*pi/180 — unit scaling, NOT trig (40.11).
        # The trivial 0 is the only exact case: 0 times anything is 0, scale preserved.
        ("degrees(0)", None, "0 (exact)"),
        ("radians(0)", None, "0 (exact)"),
        ("degrees(0.00)", None, "0.00 (exact)"),  # scale preserved on the exact zero
        ("radians(0.00)", None, "0.00 (exact)"),
        ("degrees(0)", "rational", "0 (exact)"),  # rational allows only the zero
        ("radians(0)", "rational", "0 (exact)"),
        ("degrees(0)", "floating-point", "0.0 (inexact)"),  # binary64 is always inexact
        ("radians(0)", "floating-point", "0.0 (inexact)"),
        # Fixed-point multiplies by the internal pi (29.3); inexact, rounded to the
        # operand's scale. degrees(1) ~ 57.2958 and radians(180) ~ 3.1416 round to a
        # whole number at scale 0.
        (
            "degrees(1)",
            None,
            "57 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 57.2958)",
        ),
        (
            "radians(180)",
            None,
            "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 3.1416)",
        ),
        (
            "radians(90.000000)",  # pi/2 at scale 6
            None,
            "1.570796 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.5707963268)",
        ),
        (
            "degrees(3.141593)",  # ~ pi radians -> ~ 180 degrees
            None,
            "180.000020 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 180.0000198478)",
        ),
        # Negatives convert with the sign carried through (no domain limit, every real).
        (
            "degrees(-1.000000)",
            None,
            "-57.295780 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -57.2957795131)",
        ),
        (
            "radians(-180.000000)",
            None,
            "-3.141593 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -3.1415926536)",
        ),
        # binary64 uses math.degrees / math.radians, unconditionally inexact.
        ("degrees(1)", "floating-point", "57.29577951308232 (inexact)"),
        ("radians(180)", "floating-point", "3.141592653589793 (inexact)"),
    ],
)
def test_degrees_and_radians(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        # pi has no rational value, so any non-zero argument refuses (exact-or-refuse);
        # NB no domain refusal — every real converts, the zero alone stays exact.
        ("degrees(1)", "degrees of a non-zero rational is irrational (pi)"),
        ("radians(180)", "radians of a non-zero rational is irrational (pi)"),
    ],
)
def test_degrees_and_radians_refuse_a_non_zero_rational(expression, error):
    payload = _calc(expression, "rational")
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # log(1) = 0 is the one exact case in every mode (transcendental otherwise);
        # `ln` is the alias, so ln(1) matches.
        ("log(1)", None, "0 (exact)"),
        ("ln(1)", None, "0 (exact)"),
        ("log(1)", "rational", "0 (exact)"),
        ("log(1)", "floating-point", "0.0 (inexact)"),
        # binary64 uses math.log (natural) and is unconditionally inexact.
        ("log(2)", "floating-point", "0.6931471805599453 (inexact)"),
        ("log(10)", "floating-point", "2.302585092994046 (inexact)"),
        ("log(0.5)", "floating-point", "-0.6931471805599453 (inexact)"),
        # Fixed-point reduces in base 10 and sums the atanh series at the operand's
        # scale, inexact; log(2) ~ 0.6931 rounds to 1 at scale 0 (nearest whole number).
        (
            "log(2)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.6931)",
        ),
        (
            "log(100)",  # ln(100) ~ 4.6052 rounds to 5 at scale 0
            None,
            "5 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 4.6052)",
        ),
        (
            "log(2.000000)",  # math.log(2) at scale 6
            None,
            "0.693147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6931471806)",
        ),
        # ln is the same method under its alias.
        (
            "ln(2.000000)",
            None,
            "0.693147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.6931471806)",
        ),
        (
            "log(10.000000)",  # math.log(10) at scale 6
            None,
            "2.302585 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 2.3025850930)",
        ),
        # An argument below 1 reduces to a negative power of ten -> a negative log.
        (
            "log(0.500000)",
            None,
            "-0.693147 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -0.6931471806)",
        ),
        (
            "log(0.001000)",  # ln(0.001) ~ -6.9078, heavily reduced (n = -3)
            None,
            "-6.907755 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → -6.9077552790)",
        ),
    ],
)
def test_log(expression, mode, value):
    assert _value(expression, mode) == value


_RATIONAL_LOG_REFUSAL = "logarithm of a non-unit rational is transcendental"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # No real log for a non-positive operand in any mode (x = 0 is -inf)...
        ("log(0)", None, "logarithm of a non-positive value"),
        ("log(-1)", None, "logarithm of a non-positive value"),
        ("log(0)", "floating-point", "logarithm of a non-positive value"),
        ("log(-2)", "floating-point", "logarithm of a non-positive value"),
        ("log(0)", "rational", "logarithm of a non-positive value"),
        # ...and rational refuses a transcendental log (no scale to round to).
        ("log(2)", "rational", _RATIONAL_LOG_REFUSAL),
        ("ln(2)", "rational", _RATIONAL_LOG_REFUSAL),
    ],
)
def test_log_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # A power of ten logs EXACTLY in every exact mode — the base-10 reduction
        # extracts the whole exponent with no rounding (log10's richer landmark).
        ("log10(1)", None, "0 (exact)"),
        ("log10(100)", None, "2 (exact)"),
        ("log10(1000)", None, "3 (exact)"),
        ("log10(0.010)", None, "-2.000 (exact)"),  # negative exponent, scale preserved
        ("log10(0.001000)", None, "-3.000000 (exact)"),
        ("log10(100)", "rational", "2 (exact)"),
        ("log10(1)", "rational", "0 (exact)"),
        ("log10(1/10)", "rational", "-1 (exact)"),  # 10**-1 as 1/denominator
        ("log10(1/100)", "rational", "-2 (exact)"),
        # binary64 uses math.log10 and is unconditionally inexact, even on a power of ten.
        ("log10(100)", "floating-point", "2.0 (inexact)"),
        ("log10(2)", "floating-point", "0.3010299956639812 (inexact)"),
        ("log10(0.001)", "floating-point", "-3.0 (inexact)"),
        # A non-power-of-ten fixed-point argument divides ln(x)/ln(10), inexact;
        # log10(2) ~ 0.3010 rounds to 0 at scale 0 (nearest whole number).
        (
            "log10(2)",
            None,
            "0 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 0.3010)",
        ),
        (
            "log10(2.000000)",  # math.log10(2) at scale 6
            None,
            "0.301030 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.3010299957)",
        ),
        (
            "log10(50.000000)",  # ~1.6990, between two powers of ten
            None,
            "1.698970 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 1.6989700043)",
        ),
    ],
)
def test_log10(expression, mode, value):
    assert _value(expression, mode) == value


_RATIONAL_LOG10_REFUSAL = "base-10 logarithm of a non-power-of-ten rational is irrational"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # Same non-positive refusal as log, in every mode...
        ("log10(0)", None, "logarithm of a non-positive value"),
        ("log10(-5)", None, "logarithm of a non-positive value"),
        ("log10(0)", "floating-point", "logarithm of a non-positive value"),
        ("log10(0)", "rational", "logarithm of a non-positive value"),
        # ...and rational refuses anything that is not an integer power of ten.
        ("log10(2)", "rational", _RATIONAL_LOG10_REFUSAL),
        ("log10(1/3)", "rational", _RATIONAL_LOG10_REFUSAL),
    ],
)
def test_log10_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # log2(1) = 0 is the ONLY exact landmark in every exact mode — unlike log10,
        # the base-10 reduction does NOT land powers of two exactly (28.19).
        ("log2(1)", None, "0 (exact)"),
        ("log2(1)", "rational", "0 (exact)"),
        # A power of two is NOT exact: it divides ln(x)/ln(2) like any other argument
        # and is flagged inexact even though the quotient lands on a whole number.
        (
            "log2(2)",
            None,
            "1 (inexact, rounded to 0 decimals — pass min_fixed_point_precision for more)",
        ),
        (
            "log2(8.000000)",
            None,
            "3.000000 (inexact, rounded to 6 decimals — pass min_fixed_point_precision for more)",
        ),
        # binary64 uses math.log2 and is unconditionally inexact, even on a power of two.
        ("log2(2)", "floating-point", "1.0 (inexact)"),
        ("log2(8)", "floating-point", "3.0 (inexact)"),
        ("log2(0.5)", "floating-point", "-1.0 (inexact)"),
        # A non-power-of-two fixed-point argument divides ln(x)/ln(2), inexact.
        (
            "log2(50.000000)",  # ~5.6439, between two powers of two
            None,
            "5.643856 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 5.6438561898)",
        ),
    ],
)
def test_log2(expression, mode, value):
    assert _value(expression, mode) == value


_RATIONAL_LOG2_REFUSAL = "base-2 logarithm of a non-unit rational is irrational"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # Same non-positive refusal as log, in every mode...
        ("log2(0)", None, "logarithm of a non-positive value"),
        ("log2(-5)", None, "logarithm of a non-positive value"),
        ("log2(0)", "floating-point", "logarithm of a non-positive value"),
        ("log2(0)", "rational", "logarithm of a non-positive value"),
        # ...and rational refuses anything but the unit (even a power of two like 8).
        ("log2(2)", "rational", _RATIONAL_LOG2_REFUSAL),
        ("log2(8)", "rational", _RATIONAL_LOG2_REFUSAL),
        ("log2(1/3)", "rational", _RATIONAL_LOG2_REFUSAL),
    ],
)
def test_log2_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # exp(0) = 1 is the one exact case in every mode (transcendental otherwise),
        # the inverse landmark of log(1) = 0.
        ("exp(0)", None, "1 (exact)"),
        ("exp(0)", "rational", "1 (exact)"),
        ("exp(0.000000)", None, "1.000000 (exact)"),  # scale preserved on the exact case
        ("exp(0)", "floating-point", "1.0 (inexact)"),
        # binary64 uses math.exp and is unconditionally inexact; exp(1) is e.
        ("exp(1)", "floating-point", "2.718281828459045 (inexact)"),
        ("exp(2)", "floating-point", "7.38905609893065 (inexact)"),
        ("exp(-1)", "floating-point", "0.36787944117144233 (inexact)"),
        # Fixed-point range-reduces by ln(2) and sums the all-plus series at the
        # operand's scale, inexact; exp(5) ~ 148.41 rounds to 148 at scale 0.
        (
            "exp(5)",
            None,
            "148 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 148.4132)",
        ),
        (
            "exp(1.000000)",  # e at scale 6
            None,
            "2.718282 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 2.7182818285)",
        ),
        # A negative argument is fine (k < 0): exp(-1) = 1/e, below 1.
        (
            "exp(-1.000000)",
            None,
            "0.367879 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 0.3678794412)",
        ),
        # A large argument forces many 2**k restorations; the working scale carries
        # the result's integer digits so the last place is still right.
        (
            "exp(20.000000)",
            None,
            "485165195.409790 (inexact, rounded to 6 decimals — pass min_fixed_point_precision "
            "for more; e.g. =10 → 485165195.4097902780)",
        ),
    ],
)
def test_exp(expression, mode, value):
    assert _value(expression, mode) == value


def test_exp_fixed_point_honours_a_raised_precision_floor():
    # min_fixed_point_precision lifts the scale: exp(1) to 10 places is e exactly
    # rounded, the value the scale-6 default advertises in its hint.
    assert _value("exp(1.0000000000)", None, 10) == "2.7182818285 (inexact, rounded to 10 decimals)"


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # rational refuses a transcendental exp (no scale to round to) — exp of any
        # non-zero rational is irrational (Lindemann-Weierstrass), the exact-or-refuse
        # mirror of log's non-unit refusal.
        ("exp(1)", "rational", "exp of a non-zero rational is transcendental"),
        ("exp(-2)", "rational", "exp of a non-zero rational is transcendental"),
        ("exp(1/2)", "rational", "exp of a non-zero rational is transcendental"),
    ],
)
def test_exp_rational_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


# --- nullary constants pi() and e() (29.2 / 29.3 / 29.4) --------------------
# Zero-argument calls, the one function shape that takes the run context instead
# of operands (29.2). Both are irrational, so the per-mode story is the same as
# sqrt's: float rounds its native double (inexact), fixed-point truncates to the
# run's DERIVED scale (29.3), and rational refuses outright.


@pytest.mark.parametrize(
    ("expression", "mode", "value"),
    [
        # binary64: the nearest double, carrying the inexact flag (math.pi / math.e).
        ("pi()", "floating-point", "3.141592653589793 (inexact)"),
        ("e()", "floating-point", "2.718281828459045 (inexact)"),
        # A nullary is an ordinary atom, so it threads through surrounding arithmetic.
        ("2 * pi()", "floating-point", "6.283185307179586 (inexact)"),
    ],
)
def test_nullary_floating_point(expression, mode, value):
    assert _value(expression, mode) == value


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # rational refuses: an irrational constant has no finite fraction (like sqrt(2)).
        ("pi()", "rational", "pi is irrational; no rational value"),
        ("e()", "rational", "e is irrational; no rational value"),
        ("2 * pi()", "rational", "pi is irrational; no rational value"),
    ],
)
def test_nullary_rational_refuses(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


@pytest.mark.parametrize(
    ("bare", "called"),
    [("pi", "pi()"), ("e", "e()"), ("2 * pi", "2 * pi()"), ("e ** 2", "e() ** 2")],
)
@pytest.mark.parametrize("mode", ["floating-point", "fixed-point"])
def test_bare_constant_matches_the_call_form(bare, called, mode):
    # The bare constant (29.6) is sugar for the nullary call, so it evaluates byte-for-byte
    # the same in every mode — derived fixed-point scale included (the floor of 0 here).
    assert _calc(bare, mode) == _calc(called, mode)


def test_bare_constant_rational_refuses_like_the_call():
    # Same irrational refusal as pi()/e() — the sugar changes only the syntax (29.6).
    assert _calc("pi", "rational")["error"] == "pi is irrational; no rational value"
    assert _calc("e", "rational")["error"] == "e is irrational; no rational value"


@pytest.mark.parametrize(
    ("expression", "floor", "value"),
    [
        # No literal to derive a scale from: the nullary sits at the default floor of
        # 0 decimals, the hint nudging toward min_fixed_point_precision for more (29.3).
        (
            "pi()",
            None,
            "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 3.1415)",
        ),
        (
            "e()",
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision "
            "for more; e.g. =4 → 2.7182)",
        ),
        # A 0xffff@8 literal pushes the derived scale to 8; the nullary matches it (29.3).
        (
            "0xffff@8 + pi()",
            None,
            "3.14224800 (inexact, rounded to 8 decimals — pass min_fixed_point_precision "
            "for more; e.g. =12 → 3.142248003589)",
        ),
        # A 2.000000000 literal (9 written decimals) pushes the scale to 9.
        (
            "2.000000000 + pi()",
            None,
            "5.141592653 (inexact, rounded to 9 decimals — pass min_fixed_point_precision "
            "for more; e.g. =13 → 5.1415926535897)",
        ),
        # With no literal, the floor IS the derived scale: min_fixed_point_precision=18
        # gives the bare nullary 18 decimals — the ERC-20 idiom — the floor winning
        # where it exceeds the default 0.
        ("pi()", 18, "3.141592653589793238 (inexact, rounded to 18 decimals)"),
        ("e()", 18, "2.718281828459045235 (inexact, rounded to 18 decimals)"),
    ],
)
def test_nullary_fixed_point_precision_derivation(expression, floor, value):
    # fixed-point is the default mode (mode omitted), so this also exercises the
    # mode-without-operand dispatch (29.2): the run mode reaches the nullary.
    assert _value(expression, floor=floor) == value


# --- nullary clock reading time() (28.1) ------------------------------------
# The first clock-DEPENDENT nullary. UNLIKE the irrational constants pi()/e() a
# tick is exactly rational, so it refuses in NO mode and is inexact only in float.
# The tool path samples the LIVE realtime clock (asserted only for type/range);
# the exact per-mode and per-scale renders are pinned by injecting a fixed epoch
# through evaluate()'s now_ns hook (28.1.2).

_EPOCH_NS = 1_700_000_000_123_456_789  # a fixed instant, 1700000000.123456789 s


def _time_value(expression, mode, floor=0):
    """Evaluate `expression` at the fixed test epoch, returning the Value (28.1.2)."""
    return parse(expression).evaluate(mode, min_fixed_point_precision=floor, now_ns=_EPOCH_NS)


@pytest.mark.parametrize(
    ("floor", "mantissa", "decimals"),
    [
        # Derived scale s == the floor here (no literal). The ns reading is TRUNCATED
        # to s decimals — a resolution choice, so the render is EXACT (28.1.1).
        (0, 1_700_000_000, 0),  # whole seconds, the C library time()
        (3, 1_700_000_000_123, 3),  # tv_nsec truncated to milliseconds
        (9, 1_700_000_000_123_456_789, 9),  # the full ns reading, no truncation
        (12, 1_700_000_000_123_456_789_000, 12),  # past 9: true zeros, the clock has no finer grain
    ],
)
def test_time_fixed_point_truncates_to_the_derived_scale(floor, mantissa, decimals):
    value = _time_value("time()", Mode.FIXED_POINT, floor=floor)
    assert value.mode is Mode.FIXED_POINT
    assert value.payload == FixedPoint(mantissa, decimals)
    assert value.exact is True  # the rendered decimal IS the sampled value


@pytest.mark.parametrize(
    ("floor", "rendered"),
    [
        # The headline behaviour: the precision floor IS the clock resolution, and
        # the rendered fixed-point decimal carries it verbatim (28.1.1). _EPOCH_NS is
        # 1700000000.123456789 s, so each scale is that epoch cut at that resolution.
        (0, "1700000000"),  # whole seconds
        (3, "1700000000.123"),  # milliseconds
        (6, "1700000000.123456"),  # microseconds — TRUNCATED (rounding would give ...457)
        (9, "1700000000.123456789"),  # nanoseconds — the clock's full grain
        (12, "1700000000.123456789000"),  # beyond ns: true trailing zeros, no finer grain
    ],
)
def test_time_fixed_point_renders_epoch_at_the_requested_resolution(floor, rendered):
    # The fixed-point string a caller actually sees: epoch.<fraction> at the scale
    # set by min_fixed_point_precision (3 -> ms, 6 -> us, 9 -> ns).
    assert _time_value("time()", Mode.FIXED_POINT, floor=floor).to_string() == rendered


def test_time_floating_point_is_the_full_ns_as_an_inexact_double():
    value = _time_value("time()", Mode.FLOATING_POINT)
    assert value.mode is Mode.FLOATING_POINT
    assert value.payload == float(Fraction(_EPOCH_NS, 10**9))
    assert value.exact is False  # the only mode that rounds


def test_time_rational_is_the_full_ns_exact_and_does_not_refuse():
    # Where pi()/e() raise in rational, time() returns the exact fraction (28.1).
    value = _time_value("time()", Mode.RATIONAL)
    assert value.mode is Mode.RATIONAL
    assert value.payload == Fraction(_EPOCH_NS, 10**9)
    assert value.exact is True


def test_time_is_one_instant_per_run():
    # Every time() in an expression shares the single sampled reading (28.1.2),
    # so a self-difference is exactly zero — even off the LIVE clock (now_ns
    # defaulted), since the whole run samples once.
    value = parse("time() - time()").evaluate(Mode.RATIONAL)
    assert value.payload == Fraction(0)


def test_time_live_clock_through_the_tool_is_a_recent_epoch():
    # The functional path samples the real clock; fixed-point default scale 0 gives
    # whole seconds, exact. Assert only a sane range (this test is time-dependent).
    rendered = _value("time()")  # e.g. "1765432100 (exact)"
    seconds, annotation = rendered.split(" ", 1)
    assert annotation == "(exact)"
    assert int(seconds) > 1_700_000_000  # after 2023-11-14


def test_time_rational_through_the_tool_does_not_refuse():
    # Contrast with test_nullary_rational_refuses: a clock tick has a rational value.
    payload = _calc("time()", "rational")
    assert payload["error"] is None
    assert payload["value"] is not None


# --- integral(expr, var, a, b): the definite integral special form (40.18) ----
# A SOLVER-ADJACENT special form: `expr` is the unevaluated integrand and `var` a bare
# NAME, not values. Adaptive-Simpson quadrature in the active mode (no float shadow); the
# result is ALWAYS inexact (a quadrature only approximates the integral). Simpson is EXACT
# for polynomials up to degree 3, so a low-degree integrand lands on the true value in
# every mode — still flagged inexact — which keeps these assertions deterministic.
@pytest.mark.parametrize(
    ("expression", "mode", "floor", "value"),
    [
        # Linear is Simpson-exact even at fixed-point scale 0 (midpoints stay on the grid).
        (
            "integral(x, x, 0, 2)",
            None,
            None,
            "2 (inexact, rounded to 0 decimals — pass min_fixed_point_precision for more)",
        ),
        # Raising the fixed-point floor carries the precision through the quadrature.
        ("integral(x, x, 0, 2)", None, 4, "2.0000 (inexact, rounded to 4 decimals)"),
        ("integral(x**2, x, 0, 3)", None, 4, "9.0000 (inexact, rounded to 4 decimals)"),
        # x**2 is degree 2 — Simpson-exact — so float and rational hit 9 exactly (inexact).
        ("integral(x**2, x, 0, 3)", "floating-point", None, "9.0 (inexact)"),
        ("integral(x**2, x, 0, 3)", "rational", None, "9 (inexact)"),
        ("integral(2*x+1, x, 0, 1)", "floating-point", None, "2.0 (inexact)"),
        # A constant integrand: var need not occur; the area is just height * width.
        ("integral(1, x, 0, 5)", "rational", None, "5 (inexact)"),
        # a > b integrates with sign; a == b is zero — both still flagged inexact.
        ("integral(x**2, x, 3, 0)", "rational", None, "-9 (inexact)"),
        ("integral(x, x, 2, 2)", "rational", None, "0 (inexact)"),
    ],
)
def test_integral(expression, mode, floor, value):
    assert _value(expression, mode, floor) == value


def test_integral_reads_outer_variables():
    # The integrand re-evaluates in a child store seeded from the run's, so it can read an
    # outer binding (k) while the integration variable (x) shadows the per-sample point.
    assert _value("k = 3\nintegral(k*x, x, 0, 2)", "rational") == "6 (inexact)"


@pytest.mark.parametrize(
    ("expression", "mode", "floor", "expected"),
    [
        # A transcendental integrand: not Simpson-exact, so the result only approximates —
        # assert closeness rather than an exact string. integral of sin over [0, pi] = 2.
        ("integral(sin(x), x, 0, pi)", "floating-point", None, 2.0),
        ("integral(sin(x), x, 0, pi)", None, 6, 2.0),
    ],
)
def test_integral_transcendental_is_close(expression, mode, floor, expected):
    rendered = _value(expression, mode, floor)
    number, annotation = rendered.split(" ", 1)
    assert annotation.startswith("(inexact")
    assert abs(float(number) - expected) < 1e-4


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # The 2nd argument must be a bare name: a literal or a constant nullary (pi/e) is not.
        ("integral(x, 5, 0, 1)", None, "integral's variable (2nd argument) must be a name"),
        ("integral(x, pi, 0, 1)", None, "integral's variable (2nd argument) must be a name"),
        # A transcendental integrand in rational mode refuses at the sample, like sin does
        # everywhere — surfaced as the integrand's own line-tagged error, not a special one.
        ("integral(sin(x), x, 0, 1)", "rational", "sine of a non-zero rational is irrational"),
    ],
)
def test_integral_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_integral_arity_is_a_parse_error():
    # Fixed arity 4, wired through FUNCTION_ARITIES like every call — wrong count is caught
    # at parse, before evaluation, the same as a misused ordinary function.
    payload = _calc("integral(x, x, 0)")
    assert payload["value"] is None
    assert "takes 4 argument(s)" in payload["error"]


def test_integral_masks_its_bound_variable_in_referenced_names():
    # The integration variable is BOUND (a dummy the form rebinds per sample), so it is not
    # a free reference — only genuinely free names (here `t`) leak out, which keeps the
    # dummy from being mistaken for a solver unknown when an integral is nested.
    assert parse("integral(x*t, x, 0, 1)").referenced_names() == frozenset({"t"})


# --- diff(expr, var, at): the numerical-derivative special form (40.17) -------
# diff's SIBLING special form (40.18 integral): `expr` is the unevaluated expression and
# `var` a bare NAME, not values. A five-point central difference in the active mode (no
# float shadow); the result is ALWAYS inexact (a finite-difference quotient only
# approximates the derivative). The stencil is EXACT for polynomials up to degree 4, so a
# low-degree expression lands on the true slope in every mode — still flagged inexact —
# which keeps these assertions deterministic.
@pytest.mark.parametrize(
    ("expression", "mode", "floor", "value"),
    [
        # Quadratic and cubic are stencil-exact, so they land on the true derivative even at
        # the integer grid (scale 0, where the step is one unit) and at a raised floor.
        (
            "diff(x**2, x, 3)",
            None,
            None,
            "6 (inexact, rounded to 0 decimals — pass min_fixed_point_precision for more)",
        ),
        ("diff(x**2, x, 3)", None, 4, "6.0000 (inexact, rounded to 4 decimals)"),
        ("diff(x**3, x, 2)", None, 4, "12.0000 (inexact, rounded to 4 decimals)"),
        # Exact-valued and flagged inexact in rational, across degrees 0..3.
        ("diff(x**2, x, 3)", "rational", None, "6 (inexact)"),
        ("diff(x**3, x, 2)", "rational", None, "12 (inexact)"),
        ("diff(2*x+1, x, 0)", "rational", None, "2 (inexact)"),
        # A constant expression: var need not occur; the slope is 0 (still inexact).
        ("diff(1, x, 0)", "rational", None, "0 (inexact)"),
        ("diff(x, x, 5)", "rational", None, "1 (inexact)"),
    ],
)
def test_diff(expression, mode, floor, value):
    assert _value(expression, mode, floor) == value


def test_diff_reads_outer_variables():
    # The expression re-evaluates in a child store seeded from the run's, so it can read an
    # outer binding (k) while the differentiation variable (x) shadows the per-sample point.
    # d/dx (k*x) = k = 3, stencil-exact in rational.
    assert _value("k = 3\ndiff(k*x, x, 2)", "rational") == "3 (inexact)"


@pytest.mark.parametrize(
    ("expression", "mode", "floor", "expected"),
    [
        # A transcendental expression: not stencil-exact, so the result only approximates —
        # assert closeness rather than an exact string. d/dx sin(x) at 1 = cos(1) ≈ 0.5403;
        # d/dx exp(x) at 1 = e ≈ 2.71828.
        ("diff(sin(x), x, 1)", "floating-point", None, 0.5403023058681398),
        ("diff(exp(x), x, 1)", "floating-point", None, 2.718281828459045),
        ("diff(sin(x), x, 1)", None, 6, 0.5403023058681398),
    ],
)
def test_diff_transcendental_is_close(expression, mode, floor, expected):
    rendered = _value(expression, mode, floor)
    number, annotation = rendered.split(" ", 1)
    assert annotation.startswith("(inexact")
    assert abs(float(number) - expected) < 1e-4


@pytest.mark.parametrize(
    ("expression", "mode", "error"),
    [
        # The 2nd argument must be a bare name: a literal or a constant nullary (pi/e) is not.
        ("diff(x, 5, 0)", None, "diff's variable (2nd argument) must be a name"),
        ("diff(x, pi, 0)", None, "diff's variable (2nd argument) must be a name"),
        # A transcendental expression in rational mode refuses at the sample, like sin does
        # everywhere — surfaced as the expression's own line-tagged error. (sin'(0) too: the
        # samples sit at ±h, non-zero rationals, so sine refuses there even though the point
        # is 0.)
        ("diff(sin(x), x, 1)", "rational", "sine of a non-zero rational is irrational"),
        ("diff(sin(x), x, 0)", "rational", "sine of a non-zero rational is irrational"),
    ],
)
def test_diff_refuses_with_a_line_tagged_error(expression, mode, error):
    payload = _calc(expression, mode)
    assert payload["error"] == error
    assert payload["value"] is None


def test_diff_arity_is_a_parse_error():
    # Fixed arity 3, wired through FUNCTION_ARITIES like every call — wrong count is caught
    # at parse, before evaluation, the same as a misused ordinary function.
    payload = _calc("diff(x, x)")
    assert payload["value"] is None
    assert "takes 3 argument(s)" in payload["error"]


def test_diff_masks_its_bound_variable_in_referenced_names():
    # The differentiation variable is BOUND (a dummy the form rebinds per sample), so it is
    # not a free reference — only genuinely free names (here `t`) leak out, which keeps the
    # dummy from being mistaken for a solver unknown when a diff is nested.
    assert parse("diff(x*t, x, 0)").referenced_names() == frozenset({"t"})
