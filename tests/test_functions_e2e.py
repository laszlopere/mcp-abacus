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
