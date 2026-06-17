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
