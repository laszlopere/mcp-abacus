# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end functional tests for the call functions, exercised in GROUPS.

The companion to test_functions.py's per-function coverage: where that drives ONE
function across modes and arguments, this drives the WHOLE `calculate` seam with
multi-line PROGRAMS that answer several related functions at once. A program shares
one scope, so a leading assignment feeds every following line, and each bare-
expression line is echoed in the reply's `values` array — the same path a real
client takes (dispatch -> parse -> evaluate -> render). The in-process FastMCP
`mcp.call_tool` path is used (no subprocess); test_e2e.py covers `calculate` over
real stdio.

A grouped program is the natural unit here: one call can print

    x = pi / 10
    sin(x)
    cos(x)

and get back BOTH trig values keyed to the shared `x`, so the group reads as a
self-contained worked example. Each program is framed with a `#` comment header
describing the group; the lexer strips the comments before evaluation, so the tool
sees only the code. Under ./run-tests.sh --human-readable conftest prints each call
as a framed Code / Result block (the program, then one `<expr> = <value>` line per
answered line); under `pytest -v` it prints the full REQUEST / REPLY JSON the tool
returns. Both views mirror test_solver_e2e.py.
"""

import asyncio
import json

from mcp_abacus.server import mcp


def _annotated(description: str, code: str) -> str:
    """Frame an abacus `code` snippet with a `#` comment header from `description`.

    The lexer strips the comments before evaluation, so the tool sees only the code;
    the header is what makes each printed "Code" block (and the REQUEST JSON) state
    WHAT the group computes.
    """
    header = "\n".join(f"# {line}".rstrip() for line in description.strip("\n").splitlines())
    return f"#\n{header}\n#\n\n{code}"


def _calc(program, *, mode=None, floor=None):
    """Invoke `calculate` in-process; return its structured reply dict.

    `mode` and `floor` (min_fixed_point_precision) are omitted from the request when
    None, so the tool's own defaults (fixed-point, no floor) are exercised.
    """
    arguments = {"expression": program}
    if mode is not None:
        arguments["mode"] = mode
    if floor is not None:
        arguments["min_fixed_point_precision"] = floor
    result = asyncio.run(mcp.call_tool("calculate", arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def _values(payload):
    """The annotated `value` string of every answered line, in source order.

    Fails loudly if the program errored — the per-line breakdown is null then, so a
    silent empty list would hide the failure.
    """
    assert payload["error"] is None, f"program errored: {payload['error']}"
    return [entry["value"] for entry in payload["values"]]


# --- fixed-point: the exact-grid groups (default mode) ------------------------


def test_rounding_family_agrees_and_diverges_on_a_tie():
    # floor/ceil/round/trunc on one value, fed by a shared `x`. The point of the group
    # is the half-way tie: round goes to EVEN (2, not 3), while floor and trunc agree at
    # 2 and ceil climbs to 3 — all exact on the fixed-point grid.
    program = _annotated(
        "the four roundings of 2.5 — round ties to even (2), not up",
        "x = 2.5\nfloor(x)\nceil(x)\nround(x)\ntrunc(x)",
    )
    assert _values(_calc(program)) == [
        "2 (exact)",  # floor: toward -inf
        "3 (exact)",  # ceil: toward +inf
        "2 (exact)",  # round: nearest, tie -> even
        "2 (exact)",  # trunc: toward zero
    ]


def test_roots_land_on_the_grid():
    # sqrt and cbrt where the root is exact at the operand's scale: a perfect square,
    # a half that fits scale 2, a perfect cube, and a negative cube (odd root carries
    # the sign). Every line is exact — no rounding hint.
    program = _annotated(
        "roots that land exactly on the fixed-point grid",
        "sqrt(16)\nsqrt(2.25)\ncbrt(27)\ncbrt(-8)",
    )
    assert _values(_calc(program)) == [
        "4 (exact)",
        "1.50 (exact)",  # scale 2 preserved
        "3 (exact)",
        "-2 (exact)",  # odd root keeps the sign
    ]


def test_aggregates_over_one_dataset():
    # The variadic statistics over a single dataset 2,4,6: sum totals, avg divides
    # evenly, max/min/median SELECT an operand verbatim, product multiplies. All exact
    # because the set divides cleanly at scale 0.
    program = _annotated(
        "the aggregate family over the dataset 2, 4, 6",
        "sum(2, 4, 6)\navg(2, 4, 6)\nmax(2, 4, 6)\nmin(2, 4, 6)\nmedian(2, 4, 6)\nproduct(2, 4, 6)",
    )
    assert _values(_calc(program)) == [
        "12 (exact)",
        "4 (exact)",  # (2+4+6)/3 divides evenly
        "6 (exact)",
        "2 (exact)",
        "4 (exact)",  # the middle of the sorted set
        "48 (exact)",
    ]


def test_gcd_over_integers_drops_sign_and_folds_zero():
    # The variadic gcd over integer operands: it folds across three, reduces to the
    # single-operand identity gcd(a) = |a|, DROPS the sign (works on magnitudes), and
    # treats a 0 operand as the identity (gcd(0, n) = n). All exact — pure integer math
    # on the fixed-point grid, even from .00-scaled literals that are whole-valued.
    program = _annotated(
        "gcd folds magnitudes, ignores sign, and absorbs a zero operand",
        "gcd(54, 24, 6)\ngcd(-12, 8)\ngcd(42)\ngcd(0, 5)\ngcd(0, 0)\ngcd(12.00, 8.00)",
    )
    assert _values(_calc(program)) == [
        "6 (exact)",  # gcd(54, 24, 6)
        "4 (exact)",  # sign dropped: gcd(12, 8)
        "42 (exact)",  # single-operand identity gcd(a) = |a|
        "5 (exact)",  # gcd(0, n) = n
        "0 (exact)",  # gcd(0, 0) = 0
        "4 (exact)",  # whole-valued .00 literals are integer-domain
    ]


def test_gcd_in_floating_point_is_flagged_inexact():
    # gcd is pure integer math, but float labels every result inexact (binary64 carries
    # no exactness here) — so even the whole-number gcd reads as a .0 inexact double,
    # the same convention sum/avg follow in this mode.
    program = _annotated(
        "gcd in floating-point: integer value, inexact flag",
        "gcd(54, 24, 6)\ngcd(-12, 8)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "6.0 (inexact)",
        "4.0 (inexact)",
    ]


def test_gcd_refuses_a_non_integer_operand():
    # gcd's DOMAIN is integer-valued operands only (the exact-or-refuse stance): a
    # fixed-point value with a fractional part aborts the whole program with a plain,
    # self-contained reason — the earlier exact gcd is discarded with it.
    program = _annotated(
        "a fractional operand mid-group aborts gcd",
        "gcd(12, 8)\ngcd(2.5, 5)",
    )
    payload = _calc(program)
    assert payload["values"] is None
    assert payload["error"] == "gcd requires integer operands"


def test_lcm_over_integers_folds_and_absorbs_zero():
    # The variadic lcm mirrors gcd: it folds across three operands, reduces to the
    # single-operand identity lcm(a) = |a|, DROPS the sign, and ANY zero operand makes
    # the whole result 0 (0 shares every multiple). All exact — the fold stays in the
    # integers — including from whole-valued .00 literals.
    program = _annotated(
        "lcm folds magnitudes, ignores sign, and collapses to 0 on a zero operand",
        "lcm(2, 3, 4)\nlcm(-4, 6)\nlcm(7)\nlcm(0, 5)\nlcm(0, 0)\nlcm(4.00, 6.00)",
    )
    assert _values(_calc(program)) == [
        "12 (exact)",  # lcm(2, 3, 4)
        "12 (exact)",  # sign dropped: lcm(4, 6)
        "7 (exact)",  # single-operand identity lcm(a) = |a|
        "0 (exact)",  # any zero operand -> 0
        "0 (exact)",  # lcm(0, 0) = 0
        "12 (exact)",  # whole-valued .00 literals are integer-domain
    ]


def test_gcd_and_lcm_satisfy_the_product_identity():
    # The reason to group: two legs feed both functions through one shared scope, and
    # gcd(a, b) * lcm(a, b) == |a * b| — the classic identity. With a = 12, b = 18 the
    # gcd is 6, the lcm 36, and both products equal 216. All exact integer math.
    program = _annotated(
        "gcd(a,b) * lcm(a,b) equals a*b through one shared scope",
        "a = 12\nb = 18\ngcd(a, b)\nlcm(a, b)\ngcd(a, b) * lcm(a, b)\na * b",
    )
    assert _values(_calc(program)) == [
        "6 (exact)",  # gcd(12, 18)
        "36 (exact)",  # lcm(12, 18)
        "216 (exact)",  # gcd * lcm
        "216 (exact)",  # == a * b
    ]


def test_lcm_refuses_a_non_integer_operand():
    # lcm shares gcd's integer-only DOMAIN: a fixed-point value with a fractional part
    # aborts the whole program with the parallel self-contained reason — the earlier
    # exact lcm is discarded with it.
    program = _annotated(
        "a fractional operand mid-group aborts lcm",
        "lcm(4, 6)\nlcm(2.5, 5)",
    )
    payload = _calc(program)
    assert payload["values"] is None
    assert payload["error"] == "lcm requires integer operands"


def test_population_spread_on_the_textbook_set():
    # variance and its square root stddev on the textbook set with mean 5, squared-
    # deviation sum 32: /8 = 4, and sqrt(4) = 2 is a perfect square, so both stay exact.
    program = _annotated(
        "population variance and stddev of 2,4,4,4,5,5,7,9 (mean 5)",
        "variance(2, 4, 4, 4, 5, 5, 7, 9)\nstddev(2, 4, 4, 4, 5, 5, 7, 9)",
    )
    assert _values(_calc(program)) == ["4 (exact)", "2 (exact)"]


def test_power_and_absolute_value():
    # pow as the call form of **: an integer exponent is exact, and a 1/2 exponent that
    # is a perfect root (sqrt 4 = 2) stays exact. abs drops a sign, exact from a literal
    # or a subexpression.
    program = _annotated(
        "pow (integer and exact-root exponents) and abs",
        "pow(2, 10)\npow(4, 0.5)\nabs(-7)\nabs(3 - 8)",
    )
    assert _values(_calc(program)) == [
        "1024 (exact)",
        "2.0 (exact)",  # sqrt(4), exact root
        "7 (exact)",
        "5 (exact)",  # abs of the negative subexpression 3 - 8
    ]


def test_logarithms_exact_on_powers_of_ten():
    # log10 is exact precisely on powers of ten — the one logarithm that lands on the
    # fixed-point grid. A positive and a negative power, both exact.
    program = _annotated(
        "log10 is exact on powers of ten",
        "log10(1000)\nlog10(0.001)",
    )
    assert _values(_calc(program)) == ["3 (exact)", "-3.000 (exact)"]


def test_general_logarithm_exact_on_integer_powers():
    # The two-arg log(x, base) (40.10) lands the whole exponent EXACTLY when x is an
    # integer power of the base — including negative powers and the trivial log(1, b).
    program = _annotated(
        "log(x, base): the general logarithm, exact on integer powers",
        "log(8, 2)\nlog(81, 3)\nlog(1024, 2)\nlog(1, 7)\nlog(0.25, 2)",
    )
    assert _values(_calc(program)) == [
        "3 (exact)",  # 2**3
        "4 (exact)",  # 3**4
        "10 (exact)",  # 2**10
        "0 (exact)",  # base**0 == 1 for any base
        "-2.00 (exact)",  # 2**-2 == 0.25, scale 2 from the literal
    ]


def test_general_logarithm_outdoes_log2_on_a_power_of_two():
    # log2 reduces base-10, so even log2(8) only ROUNDS to 3; the two-arg log(8, 2)
    # detects the integer power directly and is EXACT — the reason both exist.
    program = _annotated(
        "log(8, 2) is exact where log2(8) merely rounds",
        "log2(8)\nlog(8, 2)",
    )
    assert _values(_calc(program)) == [
        "3 (inexact, rounded to 0 decimals — pass min_fixed_point_precision for more)",
        "3 (exact)",
    ]


def test_general_logarithm_inexact_off_a_power_carries_the_precision_hint():
    # Off an integer power the result is transcendental: it rounds to the operand's
    # scale and flags inexact. Pinned at scale 10 (a floor) so the digits are stable.
    program = _annotated(
        "log(10, 2) and log(100, 3) — transcendental, rounded at scale 10",
        "log(10, 2)\nlog(100, 3)",
    )
    assert _values(_calc(program, floor=10)) == [
        "3.3219280949 (inexact, rounded to 10 decimals)",  # log2(10)
        "4.1918065486 (inexact, rounded to 10 decimals)",  # log3(100)
    ]


def test_general_logarithm_in_rational_is_exact_or_refuses():
    # Rational is exact-or-refuse: an integer-power landmark returns the exponent as a
    # Fraction; a non-power base is transcendental and refused (value is None).
    program = _annotated(
        "log(x, base) in rational: integer powers exact, else refused",
        "log(8, 2)\nlog(1 / 4, 2)",
    )
    assert _values(_calc(program, mode="rational")) == ["3 (exact)", "-2 (exact)"]

    refusal = _calc("log(10, 3)", mode="rational")
    assert refusal["values"] is None
    assert "transcendental" in refusal["error"]


def test_general_logarithm_in_floating_point_is_always_inexact():
    # binary64 carries no exactness: even log(8, 2) and log(100, 10), whole-number
    # results, are flagged inexact in floating-point — the mode's blanket rule.
    program = _annotated(
        "log(x, base) in floating-point: whole results still inexact",
        "log(8, 2)\nlog(100, 10)",
    )
    assert _values(_calc(program, mode="floating-point")) == ["3.0 (inexact)", "2.0 (inexact)"]


def test_general_logarithm_refuses_a_bad_base_or_argument():
    # DOMAIN: x > 0 AND base > 0, base != 1. Each refusal is line-tagged with no values.
    for code, fragment in [
        ("log(8, 1)", "base must be positive and not 1"),
        ("log(8, -2)", "base must be positive and not 1"),
        ("log(8, 0)", "base must be positive and not 1"),
        ("log(-8, 2)", "non-positive value"),
    ]:
        payload = _calc(code)
        assert payload["values"] is None, code
        assert fragment in payload["error"], code


def test_variables_feed_a_group_of_functions():
    # A shared scope is the reason to group: a and b are assigned once and read by every
    # following line — the Pythagorean hypotenuse of (3,4), the larger leg, and the sum.
    program = _annotated(
        "two legs feed three functions through one shared scope",
        "a = 3\nb = 4\nsqrt(a*a + b*b)\nmax(a, b)\nsum(a, b)",
    )
    assert _values(_calc(program)) == [
        "5 (exact)",  # sqrt(9 + 16) = 5
        "4 (exact)",
        "7 (exact)",
    ]


# --- fixed-point: the inexact transcendental group (carries the precision hint) --


def test_trigonometry_in_fixed_point_carries_the_precision_hint():
    # The same trig group as the float one below, but in fixed-point at scale 6: each
    # transcendental rounds to the operand's scale and flags inexact, and the value
    # string carries the min_fixed_point_precision offer (the worked example at scale+4).
    program = _annotated(
        "sin/cos/tan of 1.000000 in fixed-point — rounds, with the precision offer",
        "x = 1.000000\nsin(x)\ncos(x)\ntan(x)",
    )
    hint = "inexact, rounded to 6 decimals — pass min_fixed_point_precision for more"
    values = _values(_calc(program))
    assert values == [
        f"0.841471 ({hint}; e.g. =10 → 0.8414709848)",
        f"0.540302 ({hint}; e.g. =10 → 0.5403023059)",
        f"1.557408 ({hint}; e.g. =10 → 1.5574077247)",
    ]


# --- floating-point: the transcendental groups -------------------------------


def test_trigonometry_at_pi_over_ten():
    # The headline group (and the docstring's example): one angle x = pi/10 feeds sin,
    # cos and tan. Floating-point uses math.* directly, so the values are the IEEE-754
    # doubles, all inexact (float never claims exactness here).
    program = _annotated(
        "sine, cosine and tangent of the shared angle x = pi/10",
        "x = pi / 10\nsin(x)\ncos(x)\ntan(x)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "0.3090169943749474 (inexact)",
        "0.9510565162951535 (inexact)",
        "0.3249196962329063 (inexact)",
    ]


def test_inverse_trigonometry_returns_the_landmark_angles():
    # The arc-functions at their landmark inputs: asin(1) and acos(0) are both pi/2,
    # atan(1) is pi/4. Floating-point, so each is the inexact double.
    program = _annotated(
        "asin(1) = acos(0) = pi/2, atan(1) = pi/4",
        "asin(1)\nacos(0)\natan(1)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "1.5707963267948966 (inexact)",  # pi/2
        "1.5707963267948966 (inexact)",  # pi/2
        "0.7853981633974483 (inexact)",  # pi/4
    ]


def test_atan2_distinguishes_the_four_quadrants():
    # The reason atan2 exists: plain atan(y/x) collapses (1,1) and (-1,-1) onto the same
    # pi/4, blind to the quadrant. atan2(y, x) reads BOTH signs, so the same |y/x| = 1
    # fans out to the four diagonal angles pi/4, 3pi/4, -3pi/4, -pi/4 across (-pi, pi].
    program = _annotated(
        "atan2 over the four sign combinations of (x, y) = (+/-1, +/-1)",
        "atan2(1, 1)\natan2(1, -1)\natan2(-1, -1)\natan2(-1, 1)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "0.7853981633974483 (inexact)",  # quadrant I:  pi/4
        "2.356194490192345 (inexact)",  # quadrant II:  3pi/4
        "-2.356194490192345 (inexact)",  # quadrant III: -3pi/4
        "-0.7853981633974483 (inexact)",  # quadrant IV: -pi/4
    ]


def test_atan2_on_the_axes_reaches_the_landmark_angles():
    # The axis cases atan alone cannot do: atan2(0, x>0) = 0 (the only exact landmark),
    # atan2(y>0, 0) = pi/2 where the ratio is undefined, and atan2(0, x<0) = pi, the full
    # half-turn at the top of the (-pi, pi] range.
    program = _annotated(
        "atan2 along the axes: 0, pi/2 and pi",
        "atan2(0, 1)\natan2(1, 0)\natan2(0, -1)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "0.0 (inexact)",  # +x axis
        "1.5707963267948966 (inexact)",  # +y axis: pi/2
        "3.141592653589793 (inexact)",  # -x axis: pi
    ]


def test_atan2_in_fixed_point_is_exact_on_axis_and_rounds_off_it():
    # Fixed-point at scale 6: the +x-axis landmark atan2(0, 1) is EXACT 0, while the
    # quadrant-II diagonal atan2(1, -1) = 3pi/4 is transcendental, so it rounds to the
    # operand's scale and carries the min_fixed_point_precision offer.
    program = _annotated(
        "atan2 in fixed-point: exact 0 on the axis, rounded 3pi/4 off it",
        "atan2(0, 1)\natan2(1.000000, -1.000000)",
    )
    hint = "inexact, rounded to 6 decimals — pass min_fixed_point_precision for more"
    assert _values(_calc(program)) == [
        "0 (exact)",
        f"2.356194 ({hint}; e.g. =10 → 2.3561944902)",  # 3pi/4
    ]


def test_atan2_in_rational_stays_exact_only_on_the_positive_x_axis():
    # Rational is exact-or-refuse: atan2(0, x>=0) = 0 is the single representable angle
    # (any non-zero rational point's angle involves pi or an irrational arctan), so the
    # axis line is exact and a diagonal point aborts the program with that reason.
    program = _annotated(
        "the lone exact rational atan2, then a diagonal point that refuses",
        "atan2(0, 1)\natan2(0, 5)\natan2(1, 1)",
    )
    payload = _calc(program, mode="rational")
    assert payload["values"] is None
    assert payload["error"] == ("angle of a rational point is irrational except atan2(0, x>=0) = 0")


def test_logarithms_and_exponential_are_inverses():
    # The log/exp family in one group: log(e) and exp(0)/exp(1) show the inverse pair,
    # and log10/log2 hit clean integer results on their own bases' powers. Float, so all
    # inexact even where the value reads as a whole number.
    program = _annotated(
        "natural log/exp inverses, plus base-10 and base-2 logs on clean powers",
        "log(e)\nexp(0)\nexp(1)\nlog10(1000)\nlog2(8)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "1.0 (inexact)",  # ln(e)
        "1.0 (inexact)",  # e**0
        "2.718281828459045 (inexact)",  # e**1 == e
        "3.0 (inexact)",  # log10(10**3)
        "3.0 (inexact)",  # log2(2**3)
    ]


def test_hyperbolics_share_one_argument():
    # The headline hyperbolic group: one shared x = 1 feeds sinh, cosh and tanh, each
    # built from e**x and e**-x. Float, so the values are the IEEE-754 doubles, all
    # inexact, and they satisfy cosh**2 - sinh**2 = 1 (cosh > sinh > tanh here).
    program = _annotated(
        "hyperbolic sine, cosine and tangent of the shared x = 1",
        "x = 1\nsinh(x)\ncosh(x)\ntanh(x)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "1.1752011936438014 (inexact)",  # (e - 1/e)/2
        "1.5430806348152437 (inexact)",  # (e + 1/e)/2
        "0.7615941559557649 (inexact)",  # sinh/cosh
    ]


def test_hyperbolics_in_fixed_point_carry_the_precision_hint():
    # The same trio in fixed-point at scale 6: each is transcendental (built from exp),
    # so it rounds to the operand's scale and flags inexact with the min_fixed_point_
    # precision offer. A negative argument keeps the sign (sinh and tanh are odd).
    program = _annotated(
        "sinh/cosh/tanh of 1.000000, then sinh of a negative argument",
        "x = 1.000000\nsinh(x)\ncosh(x)\ntanh(x)\nsinh(-2.000000)",
    )
    hint = "inexact, rounded to 6 decimals — pass min_fixed_point_precision for more"
    assert _values(_calc(program)) == [
        f"1.175201 ({hint}; e.g. =10 → 1.1752011936)",
        f"1.543081 ({hint}; e.g. =10 → 1.5430806348)",
        f"0.761594 ({hint}; e.g. =10 → 0.7615941560)",
        f"-3.626860 ({hint}; e.g. =10 → -3.6268604078)",  # odd: sinh(-2) = -sinh(2)
    ]


def test_hyperbolic_zero_landmarks_are_exact():
    # The exact landmarks where the exp core need not run: sinh(0) = 0 and tanh(0) = 0,
    # while cosh(0) = 1 (cosh's even minimum). Exact on the fixed-point grid — no
    # rounding hint, the one place the hyperbolics escape inexactness.
    program = _annotated(
        "sinh(0) = tanh(0) = 0 and cosh(0) = 1, exact on the grid",
        "sinh(0)\ncosh(0.000000)\ntanh(0)",
    )
    assert _values(_calc(program)) == [
        "0 (exact)",
        "1.000000 (exact)",  # scale preserved from the 0.000000 operand
        "0 (exact)",
    ]


def test_hyperbolics_in_rational_keep_only_the_zero_landmarks():
    # Rational is exact-or-refuse: the zero landmarks sinh(0)=0, cosh(0)=1, tanh(0)=0
    # are representable, but a non-zero argument builds on exp (transcendental there),
    # so the group aborts on the first non-zero call with that reason.
    program = _annotated(
        "the exact rational landmarks, then a non-zero cosh that refuses",
        "sinh(0)\ncosh(0)\ntanh(0)\ncosh(1)",
    )
    payload = _calc(program, mode="rational")
    assert payload["values"] is None
    assert payload["error"] == "hyperbolic cosine of a non-zero rational is transcendental"


def test_inverse_hyperbolics_invert_the_forward_ones():
    # The inverse hyperbolics, each reducing to a logarithm: asinh and acosh both take 2,
    # atanh takes 0.5. Float, so all inexact. asinh(sinh) and acosh(cosh) round-trip — e.g.
    # asinh(2) here is the angle whose sinh is 2.
    program = _annotated(
        "asinh and acosh of 2, atanh of 0.5 — the log-built inverse hyperbolics",
        "asinh(2)\nacosh(2)\natanh(0.5)",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "1.4436354751788103 (inexact)",  # ln(2 + sqrt(5))
        "1.3169578969248166 (inexact)",  # ln(2 + sqrt(3))
        "0.5493061443340548 (inexact)",  # ln(3)/2
    ]


def test_inverse_hyperbolics_in_fixed_point_carry_the_precision_hint():
    # The same group in fixed-point at scale 6: each is transcendental (built from ln),
    # so it rounds and flags inexact with the precision offer. asinh and atanh are odd, so
    # negative arguments keep the sign (acosh's domain is x >= 1, so it has no negatives).
    program = _annotated(
        "asinh/acosh/atanh of positive args, then the odd negatives of asinh and atanh",
        "asinh(2.000000)\nacosh(2.000000)\natanh(0.500000)\nasinh(-3.000000)\natanh(-0.900000)",
    )
    hint = "inexact, rounded to 6 decimals — pass min_fixed_point_precision for more"
    assert _values(_calc(program)) == [
        f"1.443635 ({hint}; e.g. =10 → 1.4436354752)",
        f"1.316958 ({hint}; e.g. =10 → 1.3169578969)",
        f"0.549306 ({hint}; e.g. =10 → 0.5493061443)",
        f"-1.818446 ({hint}; e.g. =10 → -1.8184464592)",  # odd: asinh(-3) = -asinh(3)
        f"-1.472219 ({hint}; e.g. =10 → -1.4722194896)",  # odd: atanh(-0.9) = -atanh(0.9)
    ]


def test_inverse_hyperbolic_landmarks_are_exact():
    # The exact landmarks where the ln core need not run: asinh(0) = 0 and atanh(0) = 0,
    # and acosh(1) = 0 (the bottom of acosh's domain, where the radicand vanishes). Exact
    # on the fixed-point grid — the one place these escape inexactness.
    program = _annotated(
        "asinh(0) = atanh(0) = acosh(1) = 0, exact on the grid",
        "asinh(0)\nacosh(1.000000)\natanh(0)",
    )
    assert _values(_calc(program)) == [
        "0 (exact)",
        "0.000000 (exact)",  # scale preserved from the 1.000000 operand
        "0 (exact)",
    ]


def test_inverse_hyperbolics_in_rational_keep_only_the_landmarks():
    # Rational is exact-or-refuse: the landmarks asinh(0)=0, acosh(1)=0, atanh(0)=0 are
    # representable, but any other in-domain argument reduces to a transcendental log, so
    # the group aborts on the first such call with that reason.
    program = _annotated(
        "the exact rational landmarks, then an asinh that refuses",
        "asinh(0)\nacosh(1)\natanh(0)\nasinh(2)",
    )
    payload = _calc(program, mode="rational")
    assert payload["values"] is None
    assert payload["error"] == "inverse hyperbolic sine of a non-zero rational is transcendental"


def test_inverse_hyperbolics_refuse_outside_their_domains():
    # The domain edges, like sqrt's negative refusal: acosh is undefined below 1 and atanh
    # outside the open (-1, 1) (x = +/-1 would be +/-inf). Each aborts with a self-contained
    # domain message — checked independently since the first error ends a program.
    acosh_low = _annotated("acosh below its domain", "acosh(0.5)")
    assert _calc(acosh_low)["error"] == (
        "inverse hyperbolic cosine argument below the domain [1, inf)"
    )
    atanh_edge = _annotated("atanh at the domain boundary", "atanh(1.000000)")
    assert _calc(atanh_edge)["error"] == (
        "inverse hyperbolic tangent argument outside the domain (-1, 1)"
    )


def test_the_bare_constants_pi_and_e():
    # The nullary constants used bare (no parens) and inside an expression: pi, e, and
    # 2*pi. Floating-point gives the math module's doubles, all inexact.
    program = _annotated(
        "the circle constant and Euler's number, bare and in an expression",
        "pi\ne\n2 * pi",
    )
    assert _values(_calc(program, mode="floating-point")) == [
        "3.141592653589793 (inexact)",
        "2.718281828459045 (inexact)",
        "6.283185307179586 (inexact)",
    ]


# --- integral: the definite-integral special form (40.18) --------------------


def test_integral_polynomials_through_the_grid():
    # integral(expr, var, a, b) is a SPECIAL FORM: the 1st argument is the unevaluated
    # integrand and the 2nd a bare variable NAME, not values. Adaptive Simpson is exact for
    # polynomials up to degree 3, so these land on the true area — yet a quadrature only
    # APPROXIMATES, so every line is flagged inexact. The group covers a linear and a
    # quadratic integrand, a signed interval (a > b negates), and a constant integrand
    # (the variable need not occur). Pinned at scale 4 so the digits are stable.
    program = _annotated(
        "definite integrals of low-degree polynomials, including a signed and a constant",
        "integral(x, x, 0, 2)\nintegral(x**2, x, 0, 3)\n"
        "integral(x**2, x, 3, 0)\nintegral(1, x, 0, 5)",
    )
    assert _values(_calc(program, floor=4)) == [
        "2.0000 (inexact, rounded to 4 decimals)",  # ∫x dx over [0,2]
        "9.0000 (inexact, rounded to 4 decimals)",  # ∫x**2 dx over [0,3]
        "-9.0000 (inexact, rounded to 4 decimals)",  # a > b integrates with sign
        "5.0000 (inexact, rounded to 4 decimals)",  # constant integrand: var unused
    ]


def test_integral_reads_a_shared_assignment():
    # The reason to group: the integrand re-evaluates in a child scope seeded from the
    # program's, so a leading assignment (a = 3) feeds the integral while the integration
    # variable x shadows the per-sample point. ∫(3*x) dx over [0,2] = 6, exact-valued in
    # rational but still flagged inexact (the quadrature rule).
    program = _annotated(
        "an outer assignment feeds the integrand through the shared scope",
        "a = 3\nintegral(a * x, x, 0, 2)",
    )
    assert _values(_calc(program, mode="rational")) == ["6 (inexact)"]


def test_integral_in_rational_is_inexact_even_when_the_value_is_whole():
    # Contrast with the rational group below: where sqrt/cbrt stay EXACT on a clean result,
    # the integral is inexact in EVERY mode — Simpson is exact for these low-degree
    # integrands so the value is a whole fraction, but it stands for the true integral it
    # only approximates, so the honest flag is inexact (a == b gives a still-inexact 0).
    program = _annotated(
        "rational integrals: whole-valued, yet flagged inexact (a quadrature)",
        "integral(x**2, x, 0, 3)\nintegral(2 * x + 1, x, 0, 1)\nintegral(x, x, 2, 2)",
    )
    assert _values(_calc(program, mode="rational")) == [
        "9 (inexact)",
        "2 (inexact)",
        "0 (inexact)",  # zero-width interval
    ]


def test_integral_transcendental_rounds_at_the_floor():
    # A transcendental integrand is not Simpson-exact, so the result is a genuine
    # approximation that rounds to the operand scale and flags inexact. ∫sin(x) dx over
    # [0, pi] = 2; at scale 4 the quadrature lands on 1.9999.
    program = _annotated(
        "the area under one arch of sine, ∫sin(x) dx over [0, pi] = 2",
        "integral(sin(x), x, 0, pi)",
    )
    assert _values(_calc(program, floor=4)) == ["1.9999 (inexact, rounded to 4 decimals)"]


def test_integral_non_name_variable_aborts_the_group():
    # The 2nd argument MUST be a bare name (the variable to integrate over); a literal is
    # not, so the whole program aborts with a self-contained reason — the earlier exact
    # integral is discarded with it, like any mid-group domain refusal.
    program = _annotated(
        "a non-name integration variable aborts the program",
        "integral(x**2, x, 0, 3)\nintegral(x, 5, 0, 1)",
    )
    payload = _calc(program)
    assert payload["values"] is None
    assert payload["error"] == "integral's variable (2nd argument) must be a name"


# --- diff: the numerical-derivative special form (40.17) ---------------------


def test_diff_polynomials_through_the_grid():
    # diff(expr, var, at) is a SPECIAL FORM: the 1st argument is the unevaluated expression
    # and the 2nd a bare variable NAME, not values. The five-point central difference is
    # exact for polynomials up to degree 4, so these land on the true slope — yet a
    # finite-difference quotient only APPROXIMATES, so every line is flagged inexact. The
    # group covers a linear, a quadratic and a cubic expression, plus a constant (the
    # variable need not occur — its slope is 0). Pinned at scale 4 so the digits are stable.
    program = _annotated(
        "derivatives of low-degree polynomials, including a constant",
        "diff(2 * x + 1, x, 0)\ndiff(x**2, x, 3)\ndiff(x**3, x, 2)\ndiff(1, x, 0)",
    )
    assert _values(_calc(program, floor=4)) == [
        "2.0000 (inexact, rounded to 4 decimals)",  # d/dx (2x+1) = 2
        "6.0000 (inexact, rounded to 4 decimals)",  # d/dx x**2 at 3 = 6
        "12.0000 (inexact, rounded to 4 decimals)",  # d/dx x**3 at 2 = 12
        "0.0000 (inexact, rounded to 4 decimals)",  # constant: slope 0
    ]


def test_diff_reads_a_shared_assignment():
    # The reason to group: the expression re-evaluates in a child scope seeded from the
    # program's, so a leading assignment (a = 3) feeds the derivative while the
    # differentiation variable x shadows the per-sample point. d/dx (3*x) = 3, exact-valued
    # in rational but still flagged inexact (a finite-difference quotient).
    program = _annotated(
        "an outer assignment feeds the differentiated expression through the shared scope",
        "a = 3\ndiff(a * x, x, 2)",
    )
    assert _values(_calc(program, mode="rational")) == ["3 (inexact)"]


def test_diff_in_rational_is_inexact_even_when_the_value_is_whole():
    # Contrast with the rational group below: where sqrt/cbrt stay EXACT on a clean result,
    # the derivative is inexact in EVERY mode — the stencil is exact for these low-degree
    # expressions so the value is a whole fraction, but it stands for the true derivative it
    # only approximates, so the honest flag is inexact.
    program = _annotated(
        "rational derivatives: whole-valued, yet flagged inexact (a difference quotient)",
        "diff(x**2, x, 3)\ndiff(x**3, x, 2)\ndiff(x, x, 5)",
    )
    assert _values(_calc(program, mode="rational")) == [
        "6 (inexact)",
        "12 (inexact)",
        "1 (inexact)",
    ]


def test_diff_transcendental_rounds_at_the_floor():
    # A transcendental expression is not stencil-exact, so the result is a genuine
    # approximation that rounds to the operand scale and flags inexact. d/dx sin(x) at 1 =
    # cos(1) ≈ 0.5403; at scale 4 the quotient lands on 0.5405.
    program = _annotated(
        "the slope of sine at x = 1, d/dx sin(x) = cos(1) ≈ 0.5403",
        "diff(sin(x), x, 1)",
    )
    assert _values(_calc(program, floor=4)) == ["0.5405 (inexact, rounded to 4 decimals)"]


def test_diff_non_name_variable_aborts_the_group():
    # The 2nd argument MUST be a bare name (the variable to differentiate against); a literal
    # is not, so the whole program aborts with a self-contained reason — the earlier exact
    # derivative is discarded with it, like any mid-group domain refusal.
    program = _annotated(
        "a non-name differentiation variable aborts the program",
        "diff(x**2, x, 3)\ndiff(x, 5, 0)",
    )
    payload = _calc(program)
    assert payload["values"] is None
    assert payload["error"] == "diff's variable (2nd argument) must be a name"


# --- rational: exact selection and exact fractions ---------------------------


def test_rational_keeps_functions_exact_or_refuses():
    # In rational mode the functions that CAN stay exact do: sqrt of a perfect square,
    # cbrt of a ratio of perfect cubes (27/8 -> 3/2), and abs of a fraction. The group
    # is the set of calls rational answers without fabricating digits.
    program = _annotated(
        "rational stays exact where the result is a clean fraction",
        "sqrt(16)\ncbrt(27/8)\nabs(-1/3)",
    )
    assert _values(_calc(program, mode="rational")) == [
        "4 (exact)",
        "3/2 (exact)",  # cbrt(27/8): both parts perfect cubes
        "1/3 (exact)",
    ]


def test_gcd_in_rational_is_exact_on_integers_and_refuses_a_true_fraction():
    # In rational mode gcd of denominator-1 values is exact integer math (no scale to
    # round to is irrelevant — there's no rounding), but a true fraction (denominator
    # != 1) is not integer-valued and aborts the group, the same integer-only domain.
    program = _annotated(
        "rational gcd: exact over integers, then a 1/2 that refuses",
        "gcd(54, 24, 6)\ngcd(1/2, 3)",
    )
    payload = _calc(program, mode="rational")
    assert payload["values"] is None
    assert payload["error"] == "gcd requires integer operands"


def test_lcm_in_rational_is_exact_on_integers_and_refuses_a_true_fraction():
    # lcm in rational mirrors gcd: denominator-1 operands fold to an exact integer, but
    # a true fraction (denominator != 1) is not integer-valued and aborts the group on
    # the same integer-only domain.
    program = _annotated(
        "rational lcm: exact over integers, then a 1/2 that refuses",
        "lcm(2, 3, 4)\nlcm(1/2, 3)",
    )
    payload = _calc(program, mode="rational")
    assert payload["values"] is None
    assert payload["error"] == "lcm requires integer operands"


# --- a domain refusal aborts the whole group ---------------------------------


def test_a_domain_error_aborts_the_group_with_a_clean_message():
    # When any line is out of domain the program aborts: sqrt(-4) has no real root, so
    # the whole call returns no values and a plain, self-contained error — the earlier
    # sqrt(9) result is discarded with it.
    program = _annotated(
        "a negative square root mid-group aborts the whole program",
        "sqrt(9)\nsqrt(-4)\nsqrt(16)",
    )
    payload = _calc(program)
    assert payload["values"] is None
    assert payload["error"] == "square root of a negative value"
