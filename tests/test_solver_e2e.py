# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""End-to-end functional tests for the `solver` tool (TODO 31 / 33.14).

The companion to test_solver.py's unit-level coverage: where that drives the
module's pure helpers directly, this exercises the WHOLE seam — a request dict
goes in as the tool's arguments and the structured reply comes back, the same
path a real client takes (dispatch -> validate -> parse -> search -> render). The
in-process FastMCP `mcp.call_tool` path is used (no subprocess); test_e2e.py
covers the same tool over real stdio.

Each program is framed with a `#` comment header describing WHAT it searches for
and what to expect; the lexer strips the comments before evaluation, so the solver
sees only the code. Under ./run-tests.sh --human-readable conftest prints each
call as a Code / Result block (the header is what makes it self-describing); under
`pytest -v` it prints the full REQUEST / REPLY JSON the tool sends back.
"""

import asyncio
import json
import math
import time

import pytest

from mcp_abacus.server import mcp


def _annotated(description: str, code: str) -> str:
    """Frame an abacus `code` snippet with a `#` comment header from `description`.

    The lexer strips the comments before evaluation, so the solver sees only the
    code; the header is what makes each printed "Code" block (and the REQUEST JSON)
    state WHAT the program searches for and what it expects.
    """
    header = "\n".join(f"# {line}".rstrip() for line in description.strip("\n").splitlines())
    return f"#\n{header}\n#\n\n{code}"


def _solve(
    expression,
    *,
    variable=None,
    lower=None,
    upper=None,
    variables=None,
    objective=None,
    mode="floating-point",
    floor=None,
    algorithm=None,
):
    """Invoke the `solver` tool in-process; return its structured reply dict (31.8).

    Optional arguments are omitted from the request when None, so the tool's own
    defaults (objective find-root, algorithm golden-section) are exercised. Give
    EITHER variable+lower+upper (the single form) OR variables (the multiple form).
    """
    arguments = {"expression": expression}
    if variable is not None:
        arguments["variable"] = variable
    if lower is not None:
        arguments["lower"] = lower
    if upper is not None:
        arguments["upper"] = upper
    if variables is not None:
        arguments["variables"] = variables
    if objective is not None:
        arguments["objective"] = objective
    if mode is not None:
        arguments["mode"] = mode
    if floor is not None:
        arguments["min_fixed_point_precision"] = floor
    if algorithm is not None:
        arguments["algorithm"] = algorithm
    result = asyncio.run(mcp.call_tool("solver", arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def _num(annotated):
    """The bare number from an annotated reply string ("1.5 (approximate)" -> 1.5)."""
    return float(annotated.split(" (")[0])


# --- golden-section: roots and extrema over the tool seam ---------------------


def test_find_root_in_floating_point():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2]\n"
        "expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2)
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "golden-section-search"
    assert payload["mode"] == "floating-point"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    assert payload["iterations"] > 0


def test_find_root_finds_an_exact_root_on_the_fixed_point_grid():
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, so the expression\n"
        "evaluates to an EXACT zero there (grid polish, 31.7)",
        "2*x - 3",
    )
    payload = _solve(program, variable="x", lower=0, upper=3, mode="fixed-point", floor=1)
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_fixed_point_solver_defaults_the_floor_when_omitted():
    # 43.7: a fixed-point search at scale 0 would floor the variable to whole numbers and
    # miss a non-integer root (39). Rather than refuse the bare call, an omitted
    # min_fixed_point_precision DEFAULTS to 9, so the search resolves sub-unit values out
    # of the box: the same x**2 - 2 that once needed an explicit floor now finds sqrt(2).
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] in fixed-point with NO precision given\n"
        "the floor defaults to 9, so x ~ 1.414213562 is found (not refused, not floored)",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, mode="fixed-point")
    assert payload["error"] is None
    assert payload["mode"] == "fixed-point"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    # The found value is reported at the defaulted scale 9, so its verdict shows the
    # engaged floor (no "pass min_fixed_point_precision for more" steer).
    assert "pass min_fixed_point_precision" not in payload["value"]


def test_zero_config_solve_defaults_to_fixed_point_and_a_usable_floor():
    # The fully bare call — neither mode nor min_fixed_point_precision given — keeps the
    # default mode fixed-point (consistent with calculate) AND defaults the floor to 9,
    # so the simplest possible request resolves a non-integer root with no extra args.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] with NOTHING but the bracket\n"
        "default mode fixed-point + default floor 9 -> x ~ 1.414213562",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, mode=None)
    assert payload["error"] is None
    assert payload["mode"] == "fixed-point"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)


def test_explicit_floor_overrides_the_default():
    # The default only fills an OMITTED floor; an explicit value takes precedence. Solving
    # the same x**2 - 2 at an explicit floor of 2 reports the solution and value at scale 2
    # (1.41), not the default scale 9 (1.414213562) — proving the explicit floor is honoured
    # rather than being replaced by the default.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] in fixed-point at an EXPLICIT floor of 2\n"
        "the result is reported at scale 2 (x ~ 1.41), not the default scale 9",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, mode="fixed-point", floor=2)
    assert payload["error"] is None
    assert payload["precision"] == 2
    assert payload["solution"].split(" (")[0] == "1.41"


def test_fixed_point_solver_with_precision_converges_on_a_non_integer_root():
    # The same search WITH a floor gives the variable sub-integer resolution and
    # converges on sqrt(2) — the floor is exactly what the refusal above asks for.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] in fixed-point at scale 9\nexpect x ~ 1.414213562",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, mode="fixed-point", floor=9)
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)


def test_rational_solver_needs_no_precision():
    # 39.2: other modes resolve sub-unit values natively, so the floor stays optional
    # there — a rational search without precision is NOT refused and solves exactly.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] in rational, no precision\nexpect x = 3/2 exactly",
        "2*x - 3",
    )
    payload = _solve(program, variable="x", lower=0, upper=3, mode="rational")
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "3/2"


def test_find_minimum_finds_the_low_point():
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5]\nunimodal; expect x = 3, value 0 (the low point)",
        "(x - 3)**2",
    )
    payload = _solve(program, variable="x", lower=0, upper=5, objective="find-minimum")
    assert payload["error"] is None
    assert payload["objective"] == "find-minimum"
    assert _num(payload["solution"]) == pytest.approx(3.0, abs=1e-5)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)


def test_find_maximum_finds_the_high_point():
    program = _annotated(
        "find-maximum of 5 - (x - 1)**2 over [-2, 4]\nexpect x = 1, value 5 (the peak)",
        "5 - (x - 1)**2",
    )
    payload = _solve(program, variable="x", lower=-2, upper=4, objective="find-maximum")
    assert payload["error"] is None
    assert payload["objective"] == "find-maximum"
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-4)
    assert _num(payload["value"]) == pytest.approx(5.0, abs=1e-6)


def test_find_root_with_constants_set_by_assignments():
    # r, p are set by assignment; n is the free unknown. The header is a full-line
    # comment block, and each assignment also carries an inline comment — all stripped
    # by the lexer, so the solver sees only the three statements.
    program = _annotated(
        "find-root for n: compound-interest break-even over [0, 100]\n"
        "1000 * 1.05**n == 2000  ->  n = ln2 / ln1.05 ~ 14.2067",
        "r = 0.05  # annual rate\np = 1000  # principal\np * (1 + r)**n - 2000  # solve for n",
    )
    payload = _solve(program, variable="n", lower=0, upper=100)
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(14.2067, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


def test_find_root_with_constants_set_by_assignments_in_fixed_point():
    # REGRESSION (45): the same compound-interest solve in FIXED-POINT, which hung the
    # server outright. At the solver's default floor of 9 each candidate n makes
    # 1.05**n a fractional power whose exponent reduces to p ~ 3.5e9 over q = 2.5e8, and
    # pow's Path A used to materialise 1.05**p — a 4.7-billion-digit integer — before
    # testing whether the root was exact. Every engine hung alike, which is what showed
    # the fault was in `**`, not the solver. Both guards are engine-independent, so this
    # asserts on the default golden-section path.
    program = _annotated(
        "find-root for n in FIXED-POINT: compound-interest break-even over [0, 40]\n"
        "1000 * 1.05**n == 2000  ->  n = ln2 / ln1.05 ~ 14.2067. Each candidate is a\n"
        "scale-9 fractional exponent, the shape that used to hang pow's exact-root test",
        "r = 0.05\np = 1000\np * (1 + r)**n - 2000",
    )
    start = time.monotonic()
    payload = _solve(program, variable="n", lower=0, upper=40, mode="fixed-point")
    assert time.monotonic() - start < 10.0  # the bug was unbounded; the budget is loose
    assert payload["error"] is None
    assert payload["mode"] == "fixed-point"
    assert _num(payload["solution"]) == pytest.approx(14.2067, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-3


def test_find_root_reports_no_solution_when_zero_is_unreachable():
    program = _annotated(
        "find-root of x**2 + 1 over [0, 2]\n"
        "no solution: never zero; the closest |expr| is 1 at x = 0",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2)
    assert payload["solution"] is None and payload["value"] is None
    assert "No solution" in payload["error"] and "closest" in payload["error"]


def test_domain_failures_are_penalised_not_fatal():
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] (bracket dips below 0)\n"
        "the negative side raises a domain error per candidate (penalised\n"
        "+inf), yet expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4)
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_unset_constant_surfaces_as_an_eval_error():
    program = _annotated(
        "find-root of a * x - 1 over [0, 2]\n"
        "`a` is neither the unknown nor assigned: it fails at EVERY candidate,\n"
        "so it surfaces as a line-tagged eval error, not a region to avoid",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2)
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


# --- Nelder-Mead: single- and multi-variable over the tool seam ---------------


def test_nelder_mead_finds_a_two_variable_minimum():
    program = _annotated(
        "find-minimum of a 2-var paraboloid over x in [0, 5], y in [-4, 2]\n"
        "single minimum at (3, -1), value 0; the simplex walks both axes downhill",
        "(x - 3)**2 + (y + 1)**2",
    )
    payload = _solve(
        program,
        variables={"x": [0, 5], "y": [-4, 2]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-minimum"
    assert payload["algorithm"] == "nelder-mead"
    # Multivariate: the scalar solution is null; every unknown is in `solutions`.
    assert payload["solution"] is None
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == pytest.approx(3.0, abs=1e-3)
    assert found["y"] == pytest.approx(-1.0, abs=1e-3)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)
    assert payload["iterations"] > 0


def test_nelder_mead_solves_a_single_variable_root():
    program = _annotated(
        "find-root via Nelder-Mead (it minimises |expr|): x**2 - 2 over [0, 2]\n"
        "expect x = sqrt(2) ~ 1.41421",
        "x**2 - 2",
    )
    payload = _solve(program, variables={"x": [0, 2]}, algorithm="nelder-mead")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-5)
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_nelder_mead_finds_a_two_variable_maximum():
    program = _annotated(
        "find-maximum of a 2-var dome over x in [-2, 4], y in [-5, 1]\n"
        "peaks at (1, -2) with value 5",
        "5 - (x - 1)**2 - (y + 2)**2",
    )
    payload = _solve(
        program,
        variables={"x": [-2, 4], "y": [-5, 1]},
        objective="find-maximum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-maximum"
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == pytest.approx(1.0, abs=1e-3)
    assert found["y"] == pytest.approx(-2.0, abs=1e-3)
    assert _num(payload["value"]) == pytest.approx(5.0, abs=1e-6)


def test_nelder_mead_reports_no_solution_naming_the_point():
    program = _annotated(
        "find-root of x**2 + y**2 + 1 over x, y in [-1, 1]\n"
        "no solution: never zero; closest is 1 at the origin. The error\n"
        "names the full multivariate point it reached",
        "x**2 + y**2 + 1",
    )
    payload = _solve(program, variables={"x": [-1, 1], "y": [-1, 1]}, algorithm="nelder-mead")
    assert payload["solution"] is None and payload["solutions"] is None
    message = payload["error"]
    assert "No solution" in message and "x =" in message and "y =" in message


def test_nelder_mead_unset_constant_surfaces_as_an_eval_error():
    program = _annotated(
        "find-minimum of a * x + y over x, y in [0, 1]\n"
        "as with golden-section, `a` is neither an unknown nor assigned: it\n"
        "fails at every vertex and surfaces as a line-tagged eval error",
        "a * x + y",
    )
    payload = _solve(
        program,
        variables={"x": [0, 1], "y": [0, 1]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["solution"] is None and payload["solutions"] is None
    assert "undefined variable: a" in payload["error"]


# --- Brent parabolic: the single-variable optimise alternative (33.12) --------


def test_brent_parabolic_finds_a_minimum():
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5] via Brent's parabolic minimiser\n"
        "unimodal; expect x = 3, value 0 — the parabola pins a smooth low point fast",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="brent-parabolic",
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-minimum"
    assert payload["algorithm"] == "brent-parabolic"
    assert _num(payload["solution"]) == pytest.approx(3.0, abs=1e-5)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)
    assert payload["iterations"] > 0


def test_brent_parabolic_finds_a_maximum():
    program = _annotated(
        "find-maximum of 5 - (x - 1)**2 over [-2, 4] via Brent (it minimises -expr)\n"
        "expect x = 1, value 5 (the peak)",
        "5 - (x - 1)**2",
    )
    payload = _solve(
        program, variable="x", lower=-2, upper=4, objective="find-maximum", algorithm="brent"
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-maximum"
    assert payload["algorithm"] == "brent-parabolic"  # the alias resolves to canonical
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-4)
    assert _num(payload["value"]) == pytest.approx(5.0, abs=1e-6)


def test_brent_parabolic_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Brent (it minimises |expr|)\n"
        "expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="brent-parabolic")
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-parabolic"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-5)
    assert abs(_num(payload["value"])) < 1e-6
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_brent_parabolic_rejects_the_multiple_form():
    # Like golden-section, Brent is single-variable: the `variables` form needs
    # Nelder-Mead, so the request is refused with a pointer to the right algorithm.
    program = _annotated(
        "Brent asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="brent-parabolic")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


# --- ternary search: the plainest 1-D minimiser (33.6) ------------------------
# The third member of the golden-section / brent-parabolic family, sharing their harness
# (`minimise`): it cuts the bracket at its two TRISECTION points and drops the outer third
# on the worse side. Same unimodal guarantee and the same objectives, but two evaluations
# per step where golden-section reuses one, so it is the slowest of the three — the tests
# below pin that trade-off as much as the answers.


def test_ternary_finds_a_minimum():
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5] via ternary search\n"
        "unimodal; expect x = 3, value 0 — the bracket closes on the low point by thirds",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="ternary-search",
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-minimum"
    assert payload["algorithm"] == "ternary-search"
    assert _num(payload["solution"]) == pytest.approx(3.0, abs=1e-5)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)
    assert payload["iterations"] > 0


def test_ternary_finds_a_maximum():
    program = _annotated(
        "find-maximum of 5 - (x - 1)**2 over [-2, 4] via trisection (it minimises -expr)\n"
        "expect x = 1, value 5 (the peak)",
        "5 - (x - 1)**2",
    )
    payload = _solve(
        program, variable="x", lower=-2, upper=4, objective="find-maximum", algorithm="trisection"
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-maximum"
    assert payload["algorithm"] == "ternary-search"  # the alias resolves to canonical
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-4)
    assert _num(payload["value"]) == pytest.approx(5.0, abs=1e-6)


def test_ternary_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via ternary search (it minimises |expr|)\n"
        "expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="ternary-search")
    assert payload["error"] is None
    assert payload["algorithm"] == "ternary-search"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-5)
    assert abs(_num(payload["value"])) < 1e-6
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_ternary_reaches_a_root_that_only_touches_zero():
    # The minimisers' standing advantage over the whole bracketed family: (x - pi)**2 never
    # CROSSES zero, so there is no sign change to straddle and bisection & co. refuse it.
    # Ordering |expr| needs no crossing, so ternary walks straight in. The root is
    # IRRATIONAL on purpose — an integer one would let bisection's snap polish round its
    # closest scan point onto the root and succeed anyway, hiding the difference (the same
    # reason the Newton and Halley twins of this test use pi).
    program = _annotated(
        "find-root of (x - pi)**2 over [0, 5] via ternary search — a DOUBLE root at\n"
        "x = pi that only touches zero, so no bracketer has a sign change to work with",
        "(x - pi)**2",
    )
    payload = _solve(program, variable="x", lower=0, upper=5, algorithm="ternary-search")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(math.pi, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    # The same request through a sign-change engine is refused, which is the point.
    refused = _solve(program, variable="x", lower=0, upper=5, algorithm="bisection")
    assert refused["solution"] is None
    assert "No sign change" in refused["error"]


def test_ternary_costs_more_than_golden_section_for_the_same_answer():
    # The reason ternary is not the default, pinned so the docstring's claim cannot rot.
    # Both close the same bracket to the same tolerance and land on the same answer, but
    # ternary loses twice over: it shrinks the interval only to 2/3 per step where the
    # golden split reaches 0.618, and it pays TWO program evaluations for that step where
    # golden-section reuses a point and pays one. So it takes more ITERATIONS (asserted
    # here, the only count the reply exposes) and ~2.4x the evaluations underneath.
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5] — the SAME search, two engines\n"
        "ternary shrinks by thirds at two evaluations a step; golden-section reuses one",
        "(x - 3)**2",
    )
    common = {"variable": "x", "lower": 0, "upper": 5, "objective": "find-minimum"}
    ternary = _solve(program, algorithm="ternary-search", **common)
    golden = _solve(program, algorithm="golden-section-search", **common)
    assert ternary["error"] is None and golden["error"] is None
    assert _num(ternary["solution"]) == pytest.approx(_num(golden["solution"]), abs=1e-4)
    assert ternary["iterations"] > golden["iterations"]


def test_ternary_rejects_the_multiple_form():
    # Like the rest of its family, ternary is single-variable: the `variables` form needs
    # Nelder-Mead, so the request is refused with a pointer to the right algorithm.
    program = _annotated(
        "ternary search asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="ternary-search")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


# --- newton-optimise: the gradient minimiser (33.13) --------------------------
# The fourth member of the `minimise` family and the only one of them with a derivative:
# rather than shrinking the interval it fits a parabola to the objective's local value,
# slope and curvature and jumps to that parabola's vertex, x - g'/g''. It is therefore the
# only engine in the family that CANNOT serve find-root — |expr| has a kink exactly at the
# root — and the only one restricted to extrema.


def test_newton_optimise_finds_a_minimum():
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5] via newton-optimise\n"
        "the objective IS a parabola, so one jump to its vertex lands x = 3 exactly",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="newton-optimise",
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-minimum"
    assert payload["algorithm"] == "newton-optimise"
    assert _num(payload["solution"]) == pytest.approx(3.0, abs=1e-9)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-9)


def test_newton_optimise_finds_a_maximum():
    program = _annotated(
        "find-maximum of 5 - (x - 1)**2 over [-2, 4] via newton-optimise\n"
        "the fold negates the expression, so the peak is the minimum it actually steps to",
        "5 - (x - 1)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=-2,
        upper=4,
        objective="find-maximum",
        algorithm="newton optimize",
    )
    assert payload["error"] is None
    assert payload["objective"] == "find-maximum"
    assert payload["algorithm"] == "newton-optimise"  # the alias resolves to canonical
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-9)
    assert _num(payload["value"]) == pytest.approx(5.0, abs=1e-9)


def test_newton_optimise_refuses_find_root():
    # The one objective a derivative optimiser cannot take, and the reason it is the only
    # restricted engine in its family: for find-root the folded objective is |expr|, which
    # has a KINK at the very point being sought — no curvature there to divide by. The
    # error names the engines built for a root instead.
    program = _annotated(
        "find-root of x**2 - 2 via newton-optimise — refused, it only finds EXTREMA\n"
        "(|expr| is not differentiable at the root; use newton-raphson or halley)",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-optimise")
    assert payload["solution"] is None and payload["value"] is None
    assert "only finds extrema" in payload["error"]
    assert "kink" in payload["error"]
    assert "newton-raphson" in payload["error"] and "halley" in payload["error"]


def test_newton_optimise_beats_the_derivative_free_engines():
    # What the derivative buys, on a non-quadratic where nobody gets the answer for free.
    # The bracket shrinkers must pin a FLAT optimum by interval width, which round-off
    # limits to about the square root of precision; newton-optimise instead drives the
    # SLOPE to zero, which stays sharp there. So it wins on both counts at once — fewer
    # iterations AND a closer answer. exp(x) - 3x has its minimum at ln 3.
    program = _annotated(
        "find-minimum of exp(x) - 3*x over [0, 3] — one objective, four engines\n"
        "the true minimiser is ln(3) = 1.0986122886681098",
        "exp(x) - 3*x",
    )
    common = {"variable": "x", "lower": 0, "upper": 3, "objective": "find-minimum"}
    newton = _solve(program, algorithm="newton-optimise", **common)
    brent = _solve(program, algorithm="brent-parabolic", **common)
    golden = _solve(program, algorithm="golden-section-search", **common)
    assert newton["error"] is None and brent["error"] is None and golden["error"] is None
    truth = math.log(3)
    newton_err = abs(_num(newton["solution"]) - truth)
    assert newton_err < abs(_num(brent["solution"]) - truth)
    assert newton_err < abs(_num(golden["solution"]) - truth)
    assert newton["iterations"] < brent["iterations"] < golden["iterations"]


def test_newton_optimise_stops_when_a_step_no_longer_improves():
    # An extremum is FLAT, so no method can locate it better than ~sqrt(precision): near
    # the minimum the objective stops distinguishing points and the stencil's slope is
    # mostly round-off. The step-size test alone would then let the iteration wander until
    # the 200-iteration cap, so the loop also stops the moment a step fails to improve the
    # objective. Pinned here: the search settles in a handful of steps, not hundreds.
    program = _annotated(
        "find-minimum of exp(x) - 3*x over [0, 3] via newton-optimise\n"
        "a smooth non-quadratic: converges in a few steps and does NOT run to the cap",
        "exp(x) - 3*x",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=3,
        objective="find-minimum",
        algorithm="newton-optimise",
    )
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(math.log(3), abs=1e-9)
    assert payload["iterations"] < 20  # the cap is 200; wandering would reach it


def test_newton_optimise_keeps_its_answer_inside_the_bracket():
    # The stencil samples x +- 2h, which near an endpoint would fall OUTSIDE the caller's
    # interval — and being real evaluations they would feed best-tracking, letting the
    # engine return a point the caller never asked about. The minimum of `x` over [0, 5]
    # sits exactly on the lower endpoint, the case that exposes it: the answer must be 0,
    # never the -2e-05 an uncentred stencil would otherwise reach.
    program = _annotated(
        "find-minimum of x over [0, 5] via newton-optimise — the minimum IS the endpoint\n"
        "the answer must stay inside [0, 5]; the derivative stencil must not leak past it",
        "x",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="newton-optimise",
    )
    assert payload["error"] is None
    assert _num(payload["solution"]) == 0.0
    assert _num(payload["solution"]) >= 0.0  # the point of the test: never below `lower`


def test_newton_optimise_handles_a_domain_error_region():
    # sqrt(x) is undefined below 0, so the left half of the bracket raises a domain error
    # and folds to +inf. The seed scan steers to the good half and the maximum of
    # sqrt(x) - x is found at x = 1/4, where the slope 1/(2*sqrt(x)) - 1 vanishes.
    program = _annotated(
        "find-maximum of sqrt(x) - x over [-1, 4] via newton-optimise\n"
        "the left half is a domain error (+inf); expect x = 0.25, value 0.25",
        "sqrt(x) - x",
    )
    payload = _solve(
        program,
        variable="x",
        lower=-1,
        upper=4,
        objective="find-maximum",
        algorithm="newton-optimise",
    )
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(0.25, abs=1e-6)
    assert _num(payload["value"]) == pytest.approx(0.25, abs=1e-6)


def test_newton_optimise_finds_an_exact_minimum_on_the_fixed_point_grid():
    # The grid polish applies to this engine as to every other single-unknown one, and the
    # coarse per-mode stencil step keeps the SECOND difference (which divides by 12h**2)
    # out of the quantisation noise that a scale-sized h would drown it in.
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5] via newton-optimise in fixed-point\n"
        "expect x = 3 exactly, an EXACT zero at the vertex (grid polish)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        mode="fixed-point",
        floor=9,
        algorithm="newton-optimise",
    )
    assert payload["error"] is None
    assert _num(payload["solution"]) == 3.0
    assert payload["exact"] is True


def test_newton_optimise_rejects_the_multiple_form():
    # Single-variable like the rest of its family; the multivariate gradient engine (BFGS)
    # is not in this build, so the `variables` form still points at Nelder-Mead.
    program = _annotated(
        "newton-optimise asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(
        program,
        variables={"x": [0, 1], "y": [0, 1]},
        objective="find-minimum",
        algorithm="newton-optimise",
    )
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


# --- bisection: bracketed roots via a sign change (33.1) -----------------------
# The robust single-variable ROOT finder, and the only engine that works on the raw
# SIGNED expression rather than minimising |expr|: it scans the bracket for a cell
# whose endpoints straddle zero and halves it. Find-root only (an extremum has no sign
# change to bracket), and the endpoints need NOT already straddle — the scan finds the
# first sign change inside the bracket. A function that never reaches zero gets a
# DISTINCT "no sign change" error, separate from the minimisers' "No solution".


def test_bisection_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via bisection (endpoints straddle zero:\n"
        "f(0) = -2, f(2) = 2); expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="bisection")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "bisection"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    assert payload["iterations"] > 0
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_bisection_scans_when_endpoints_share_a_sign():
    # The endpoints do NOT straddle — f(-2) = 2 and f(2) = 2 are both positive — yet a
    # root lies inside. The coarse scan finds the first sign-changing cell (scanning from
    # the lower end, that is the crossing at -sqrt(2)), so bisection still solves it and
    # returns the LEFTMOST root, where a straddle-only method would have refused outright.
    program = _annotated(
        "find-root of x**2 - 2 over [-2, 2] via bisection; both endpoints are +2,\n"
        "so the scan hunts the sign change. Leftmost root is x = -sqrt(2) ~ -1.41421",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="bisection")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(-(2**0.5), abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_bisection_finds_an_exact_root_on_the_fixed_point_grid():
    # The fixed-point grid polish applies to bisection exactly as to golden-section: the
    # halving stops within one grid step of x = 1.5, and re-testing the grid neighbours
    # lands it exactly, so the expression is an EXACT zero there.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via bisection in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program, variable="x", lower=0, upper=3, mode="fixed-point", floor=1, algorithm="bisection"
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_bisection_reports_no_sign_change():
    # x**2 + 1 is strictly positive, so it never crosses zero and the scan finds no
    # sign-changing cell anywhere. The error is DISTINCT from the minimisers' "No
    # solution": it names the missing sign change and points at the alternatives.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via bisection\n"
        "never crosses zero, so there is no sign change to bracket",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="bisection")
    assert payload["solution"] is None and payload["value"] is None
    assert "No sign change" in payload["error"]
    assert "find-minimum" in payload["error"]  # the pointer to a minimiser for a touch-root


def test_bisection_rejects_a_non_root_objective():
    # Bisection brackets a sign change of the expression itself, which only locates a
    # ROOT; an extremum has no sign change to straddle, so find-minimum is refused with
    # a pointer to the engines that do minimise.
    program = _annotated(
        "bisection asked to find-minimum — refused, it only finds roots\n"
        "(an extremum has no sign change to bracket)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="bisection",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "golden-section" in payload["error"] or "brent" in payload["error"]


def test_bisection_rejects_the_multiple_form():
    # Like the other 1-D engines, bisection is single-variable: the `variables` form
    # needs Nelder-Mead, so the request is refused with a pointer to the right algorithm.
    program = _annotated(
        "bisection asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="bisection")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_bisection_skips_domain_failures_in_the_scan():
    # The scan's left half raises a domain error (sqrt of a negative); those cells carry
    # no signed value and are skipped, yet the crossing at x = 1 on the valid side is
    # still bracketed and found — the same robustness the minimisers show for +inf.
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] via bisection (the bracket dips below 0)\n"
        "the negative side has no real value and is skipped; expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4, algorithm="bisection")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_bisection_unset_constant_surfaces_as_an_eval_error():
    # A structural failure (a constant the program never sets) fails at EVERY candidate
    # and must surface as a line-tagged eval error, not be mistaken for a domain gap the
    # scan steers around — the same contract the other engines honour.
    program = _annotated(
        "find-root of a * x - 1 over [0, 2] via bisection; `a` is never set,\n"
        "so it surfaces as an eval error, not a region to skip",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="bisection")
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


def test_bisection_root_snaps_to_an_integer():
    # The float snap polish applies to bisection too: a crossing that converges a few
    # ULPs off a clean integer is re-snapped onto it (exact `==`, not approx).
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] via bisection in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, algorithm="bisection")
    assert payload["error"] is None
    assert payload["algorithm"] == "bisection"
    assert _num(payload["solution"]) == 5.0


def test_bisection_alias_resolves_to_canonical():
    # The `bisect` spelling resolves to the canonical engine name in the reply.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the `bisect` alias\n"
        "the reply reports the canonical 'bisection'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="bisect")
    assert payload["error"] is None
    assert payload["algorithm"] == "bisection"


def test_bisection_solves_keplers_equation():
    # The transcendental-root benchmark for the 1-D bracketers, now via bisection:
    # Kepler's equation E - e*sin(E) = M is smooth and monotone in [0, pi], so a single
    # sign change brackets the eccentric anomaly cleanly.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via bisection\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="bisection")
    assert payload["error"] is None
    assert payload["algorithm"] == "bisection"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


# --- Ridders' method: superlinear bracketed roots (33.5) ----------------------
# The faster sibling of bisection: same sign-change bracket (so the same scan, find-root
# only, and distinct "no sign change" error), but each step takes Ridders' exponential-
# fit root instead of the midpoint, converging at order ~1.84 rather than linearly.


def test_ridders_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Ridders (endpoints straddle zero:\n"
        "f(0) = -2, f(2) = 2); expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="ridders")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "ridders"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    assert payload["iterations"] > 0
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_ridders_converges_in_few_iterations():
    # The point of Ridders over bisection: superlinear convergence. On a smooth root the
    # exponential fit reaches full double precision in a handful of steps, where bisection
    # would need ~50 — so the reported iteration count stays small.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Ridders; converges in a few steps\n"
        "(superlinear), not the ~50 bisection's linear halving would take",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="ridders")
    assert payload["error"] is None
    assert payload["iterations"] <= 10


def test_ridders_scans_when_endpoints_share_a_sign():
    # Like bisection, Ridders scans for the sign change, so same-sign endpoints (here
    # f(-2) = f(2) = 2) are no obstacle: it brackets and returns the leftmost root.
    program = _annotated(
        "find-root of x**2 - 2 over [-2, 2] via Ridders; both endpoints are +2,\n"
        "so the scan hunts the sign change. Leftmost root is x = -sqrt(2) ~ -1.41421",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="ridders")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(-(2**0.5), abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_ridders_finds_an_exact_root_on_the_fixed_point_grid():
    # The fixed-point grid polish applies to Ridders exactly as to bisection: the refined
    # bracket lands within one grid step of x = 1.5, and the neighbour probes pin it to
    # an EXACT zero on the grid.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via Ridders in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program, variable="x", lower=0, upper=3, mode="fixed-point", floor=1, algorithm="ridders"
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_ridders_reports_no_sign_change():
    # x**2 + 1 never crosses zero, so the scan finds no sign-changing cell: the distinct
    # "no sign change" error, naming the engine and pointing at the alternatives.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via Ridders\n"
        "never crosses zero, so there is no sign change to bracket",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="ridders")
    assert payload["solution"] is None and payload["value"] is None
    assert "No sign change" in payload["error"]
    assert "find-minimum" in payload["error"]


def test_ridders_rejects_a_non_root_objective():
    # Ridders brackets a sign change, which only locates a ROOT; find-minimum is refused
    # with a pointer to the engines that do minimise.
    program = _annotated(
        "Ridders asked to find-minimum — refused, it only finds roots\n"
        "(an extremum has no sign change to bracket)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="ridders",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "golden-section" in payload["error"] or "brent" in payload["error"]


def test_ridders_rejects_the_multiple_form():
    # Single-variable like the other 1-D engines: the `variables` form needs Nelder-Mead.
    program = _annotated(
        "Ridders asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="ridders")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_ridders_skips_domain_failures_in_the_scan():
    # The left half raises a domain error (sqrt of a negative); those cells carry no
    # signed value and are skipped, yet the crossing at x = 1 is still bracketed.
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] via Ridders (the bracket dips below 0)\n"
        "the negative side has no real value and is skipped; expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4, algorithm="ridders")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_ridders_unset_constant_surfaces_as_an_eval_error():
    # A structural failure (a constant the program never sets) fails everywhere and must
    # surface as a line-tagged eval error, not be mistaken for a domain gap to skip.
    program = _annotated(
        "find-root of a * x - 1 over [0, 2] via Ridders; `a` is never set,\n"
        "so it surfaces as an eval error, not a region to skip",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="ridders")
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


def test_ridders_root_snaps_to_an_integer():
    # The float snap polish applies to Ridders too: a crossing a few ULPs off a clean
    # integer is re-snapped onto it (exact `==`, not approx).
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] via Ridders in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, algorithm="ridders")
    assert payload["error"] is None
    assert payload["algorithm"] == "ridders"
    assert _num(payload["solution"]) == 5.0


def test_ridders_alias_resolves_to_canonical():
    # The `ridder` spelling resolves to the canonical engine name in the reply.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the `ridder` alias\n"
        "the reply reports the canonical 'ridders'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="ridder")
    assert payload["error"] is None
    assert payload["algorithm"] == "ridders"


def test_ridders_solves_keplers_equation():
    # The transcendental-root benchmark via Ridders: Kepler's equation is smooth and
    # monotone in [0, pi], so the exponential fit pins the eccentric anomaly fast.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via Ridders\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="ridders")
    assert payload["error"] is None
    assert payload["algorithm"] == "ridders"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


# --- Brent-Dekker: interpolate-or-bisect bracketed roots (33.2) ---------------
# The third engine on the sign-change bracket, and the one most libraries default to:
# inverse-quadratic interpolation through the three latest points (secant when two of
# their values coincide), accepted only inside the trusted quarter-interval and while the
# steps keep halving — otherwise a plain bisection. Same scan, same find-root-only rule,
# and the same distinct "no sign change" error as bisection / Ridders. Distinct from
# brent-parabolic (33.12), which MINIMISES; bare `brent` still names that one.


def test_brent_dekker_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Brent-Dekker (endpoints straddle zero:\n"
        "f(0) = -2, f(2) = 2); expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="brent-dekker")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "brent-dekker"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    assert payload["iterations"] > 0
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_brent_dekker_converges_in_few_iterations():
    # The point of Brent-Dekker over bisection: the interpolation is superlinear, so a
    # smooth root is pinned to full double precision in a handful of steps rather than
    # the ~50 bisection's linear halving would take.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Brent-Dekker; converges in a few steps\n"
        "(superlinear interpolation), not the ~50 of bisection's linear halving",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="brent-dekker")
    assert payload["error"] is None
    assert payload["iterations"] <= 10


def test_brent_dekker_scans_when_endpoints_share_a_sign():
    # Like its two siblings, Brent-Dekker scans for the sign change, so same-sign
    # endpoints (here f(-2) = f(2) = 2) are no obstacle: it brackets the leftmost root.
    program = _annotated(
        "find-root of x**2 - 2 over [-2, 2] via Brent-Dekker; both endpoints are +2,\n"
        "so the scan hunts the sign change. Leftmost root is x = -sqrt(2) ~ -1.41421",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="brent-dekker")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(-(2**0.5), abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_brent_dekker_finds_an_exact_root_on_the_fixed_point_grid():
    # The fixed-point grid polish applies here exactly as to bisection: the refined
    # bracket lands within one grid step of x = 1.5, and the neighbour probes pin it to
    # an EXACT zero on the grid.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via Brent-Dekker in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=3,
        mode="fixed-point",
        floor=1,
        algorithm="brent-dekker",
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_brent_dekker_reports_no_sign_change():
    # x**2 + 1 never crosses zero, so the scan finds no sign-changing cell: the distinct
    # "no sign change" error, naming the engine and pointing at the alternatives.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via Brent-Dekker\n"
        "never crosses zero, so there is no sign change to bracket",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="brent-dekker")
    assert payload["solution"] is None and payload["value"] is None
    assert "No sign change" in payload["error"]
    assert "brent-dekker" in payload["error"]
    assert "find-minimum" in payload["error"]


def test_brent_dekker_rejects_a_non_root_objective():
    # Brent-Dekker brackets a sign change, which only locates a ROOT; find-minimum is
    # refused with a pointer to the engines that do minimise — brent-PARABOLIC among them.
    program = _annotated(
        "Brent-Dekker asked to find-minimum — refused, it only finds roots\n"
        "(an extremum has no sign change to bracket; brent-parabolic is the minimiser)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="brent-dekker",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "brent-parabolic" in payload["error"]


def test_brent_dekker_rejects_the_multiple_form():
    # Single-variable like the other 1-D engines: the `variables` form needs Nelder-Mead.
    program = _annotated(
        "Brent-Dekker asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="brent-dekker")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_brent_dekker_skips_domain_failures_in_the_scan():
    # The left half raises a domain error (sqrt of a negative); those cells carry no
    # signed value and are skipped, yet the crossing at x = 1 is still bracketed.
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] via Brent-Dekker (the bracket dips below 0)\n"
        "the negative side has no real value and is skipped; expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4, algorithm="brent-dekker")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_brent_dekker_unset_constant_surfaces_as_an_eval_error():
    # A structural failure (a constant the program never sets) fails everywhere and must
    # surface as a line-tagged eval error, not be mistaken for a domain gap to skip.
    program = _annotated(
        "find-root of a * x - 1 over [0, 2] via Brent-Dekker; `a` is never set,\n"
        "so it surfaces as an eval error, not a region to skip",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="brent-dekker")
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


def test_brent_dekker_root_snaps_to_an_integer():
    # The float snap polish applies here too: a crossing a few ULPs off a clean integer
    # is re-snapped onto it (exact `==`, not approx).
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] via Brent-Dekker in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, algorithm="brent-dekker")
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-dekker"
    assert _num(payload["solution"]) == 5.0


def test_brent_dekker_alias_resolves_to_canonical():
    # The `brent-root` spelling resolves to the canonical engine name in the reply.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the `brent-root` alias\n"
        "the reply reports the canonical 'brent-dekker'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="brent-root")
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-dekker"


def test_bare_brent_still_names_the_parabolic_minimiser():
    # 33.2's naming contract, over the wire: adding Brent's ROOT method did NOT repoint
    # bare `brent`. It still reaches brent-parabolic, which serves find-minimum — a
    # find-root-only engine would have refused this call outright.
    program = _annotated(
        "find-minimum of (x - 3)**2 over [0, 5] via the bare `brent` alias\n"
        "it still names the PARABOLIC MINIMISER, so find-minimum works: expect x = 3",
        "(x - 3)**2",
    )
    payload = _solve(
        program, variable="x", lower=0, upper=5, objective="find-minimum", algorithm="brent"
    )
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-parabolic"
    assert _num(payload["solution"]) == pytest.approx(3.0, abs=1e-6)


def test_brent_dekker_solves_keplers_equation():
    # The transcendental-root benchmark via Brent-Dekker: Kepler's equation is smooth and
    # monotone in [0, pi], so the interpolation pins the eccentric anomaly fast.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via Brent-Dekker\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="brent-dekker")
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-dekker"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


def test_brent_dekker_handles_a_pathological_root():
    # Where the fallback earns its keep: a steeply decaying pole-adjacent curve over a
    # very wide bracket, on which pure interpolation overshoots. The step guard rejects
    # those proposals and bisection carries the search, so the root is still found.
    program = _annotated(
        "find-root of 1/(x - 0.5) - 4 over [0.6, 40] via Brent-Dekker\n"
        "a steeply decaying curve over a wide bracket; the interpolation is rejected\n"
        "often and bisection carries it. Expect x = 0.75, where 1/0.25 = 4",
        "1/(x - 0.5) - 4",
    )
    payload = _solve(program, variable="x", lower=0.6, upper=40, algorithm="brent-dekker")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(0.75, abs=1e-6)
    # Still far inside the iteration cap despite the interpolation being little help.
    assert payload["iterations"] < 40


def test_brent_dekker_takes_a_root_sitting_on_a_scan_node():
    # The degenerate bracket: x = 1 is exactly a scan node of [0, 2], so the scan hands
    # the loop a zero-width interval. It must recognise the root immediately (no
    # interpolation to attempt, no division by a zero-width bracket) and report it.
    program = _annotated(
        "find-root of x - 1 over [0, 2] via Brent-Dekker; x = 1 lands exactly on a\n"
        "scan node, so there is no bracket to refine — the root is taken as found",
        "x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="brent-dekker")
    assert payload["error"] is None
    assert _num(payload["solution"]) == 1.0
    assert _num(payload["value"]) == 0.0
    assert payload["iterations"] == 0


# --- Chandrupatla: interpolation under a sharper criterion (33.7) -------------
# The fourth engine on the sign-change bracket and Brent-Dekker's direct rival: the same
# inverse quadratic interpolation, but admitted by ONE geometric test on the three points
# (1 - sqrt(1-xi) < phi < sqrt(xi)) rather than Brent's chain of heuristics. Level with it
# on a simple root; markedly better on a repeated one, where the test rejects the useless
# interpolation outright and the search runs at bisection's own rate.


def test_chandrupatla_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Chandrupatla (endpoints straddle zero:\n"
        "f(0) = -2, f(2) = 2); expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatla")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "chandrupatla"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    assert payload["iterations"] > 0
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_chandrupatla_converges_in_few_iterations():
    # The interpolation is superlinear when the criterion admits it, so a smooth root is
    # pinned in a handful of steps rather than the ~35 bisection's halving would take.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Chandrupatla; converges in a few steps\n"
        "(the criterion admits the interpolation on a smooth simple root)",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatla")
    assert payload["error"] is None
    assert payload["iterations"] <= 10


def test_chandrupatla_beats_brent_dekker_on_a_repeated_root():
    # 33.7's whole claim, pinned as a comparison rather than a magic number: on the triple
    # root of x**3 the interpolation is worthless. Chandrupatla's criterion rejects it and
    # the search falls back to bisection's own rate, while Brent-Dekker's looser
    # heuristics keep accepting near-useless steps and it takes substantially longer.
    program = _annotated(
        "find-root of x**3 over [-1, 2] — a TRIPLE root at x = 0, where inverse\n"
        "quadratic interpolation is worthless. Chandrupatla's criterion rejects it\n"
        "and bisects; Brent-Dekker keeps trying and needs far more steps",
        "x**3",
    )
    chand = _solve(program, variable="x", lower=-1, upper=2, algorithm="chandrupatla")
    brent = _solve(program, variable="x", lower=-1, upper=2, algorithm="brent-dekker")
    assert chand["error"] is None and brent["error"] is None
    # Both must still land the root exactly — speed is never bought with accuracy.
    assert _num(chand["solution"]) == 0.0
    assert _num(brent["solution"]) == 0.0
    assert chand["iterations"] < brent["iterations"]


def test_chandrupatla_scans_when_endpoints_share_a_sign():
    # Like the rest of the family, Chandrupatla scans for the sign change, so same-sign
    # endpoints (here f(-2) = f(2) = 2) are no obstacle: it brackets the leftmost root.
    program = _annotated(
        "find-root of x**2 - 2 over [-2, 2] via Chandrupatla; both endpoints are +2,\n"
        "so the scan hunts the sign change. Leftmost root is x = -sqrt(2) ~ -1.41421",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="chandrupatla")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(-(2**0.5), abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_chandrupatla_finds_an_exact_root_on_the_fixed_point_grid():
    # The fixed-point grid polish applies here exactly as to bisection: the refined
    # bracket lands within one grid step of x = 1.5, and the neighbour probes pin it to
    # an EXACT zero on the grid.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via Chandrupatla in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=3,
        mode="fixed-point",
        floor=1,
        algorithm="chandrupatla",
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_chandrupatla_reports_no_sign_change():
    # x**2 + 1 never crosses zero, so the scan finds no sign-changing cell: the distinct
    # "no sign change" error, naming the engine and pointing at the alternatives.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via Chandrupatla\n"
        "never crosses zero, so there is no sign change to bracket",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="chandrupatla")
    assert payload["solution"] is None and payload["value"] is None
    assert "No sign change" in payload["error"]
    assert "chandrupatla" in payload["error"]
    assert "find-minimum" in payload["error"]


def test_chandrupatla_rejects_a_non_root_objective():
    # It brackets a sign change, which only locates a ROOT; find-minimum is refused with
    # a pointer to the engines that do minimise.
    program = _annotated(
        "Chandrupatla asked to find-minimum — refused, it only finds roots\n"
        "(an extremum has no sign change to bracket)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="chandrupatla",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "brent-parabolic" in payload["error"]


def test_chandrupatla_rejects_the_multiple_form():
    # Single-variable like the other 1-D engines: the `variables` form needs Nelder-Mead.
    program = _annotated(
        "Chandrupatla asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="chandrupatla")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_chandrupatla_skips_domain_failures_in_the_scan():
    # The left half raises a domain error (sqrt of a negative); those cells carry no
    # signed value and are skipped, yet the crossing at x = 1 is still bracketed.
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] via Chandrupatla (the bracket dips below 0)\n"
        "the negative side has no real value and is skipped; expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4, algorithm="chandrupatla")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_chandrupatla_unset_constant_surfaces_as_an_eval_error():
    # A structural failure (a constant the program never sets) fails everywhere and must
    # surface as a line-tagged eval error, not be mistaken for a domain gap to skip.
    program = _annotated(
        "find-root of a * x - 1 over [0, 2] via Chandrupatla; `a` is never set,\n"
        "so it surfaces as an eval error, not a region to skip",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatla")
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


def test_chandrupatla_root_snaps_to_an_integer():
    # The float snap polish applies here too: a crossing a few ULPs off a clean integer
    # is re-snapped onto it (exact `==`, not approx).
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] via Chandrupatla in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, algorithm="chandrupatla")
    assert payload["error"] is None
    assert payload["algorithm"] == "chandrupatla"
    assert _num(payload["solution"]) == 5.0


def test_chandrupatla_takes_a_root_sitting_on_a_scan_node():
    # x = 1 is exactly a scan node of [0, 2], so the cell before it straddles with an
    # endpoint value of precisely zero. One step and the fm == 0 test ends the search.
    # (Unlike brent-dekker's 0 here: this loop samples first and tests after, so a root
    # already in hand still costs the one evaluation. A structural difference, not a miss.)
    program = _annotated(
        "find-root of x - 1 over [0, 2] via Chandrupatla; x = 1 lands exactly on a\n"
        "scan node, so the search ends as soon as it sees the zero residual",
        "x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatla")
    assert payload["error"] is None
    assert _num(payload["solution"]) == 1.0
    assert _num(payload["value"]) == 0.0
    assert payload["iterations"] <= 2


def test_chandrupatla_handles_a_zero_width_bracket():
    # The truly degenerate case: the root IS the first scan node, so the scan hands the
    # loop a == b — a zero-width bracket. The criterion divides by that width (the `tl`
    # term), so the engine must bail out BEFORE dividing rather than raise ZeroDivisionError.
    program = _annotated(
        "find-root of x over [0, 2] via Chandrupatla; the root x = 0 is the FIRST\n"
        "scan node, so the bracket handed to the loop has zero width — it must not divide by it",
        "x",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatla")
    assert payload["error"] is None
    assert _num(payload["solution"]) == 0.0
    assert _num(payload["value"]) == 0.0
    assert payload["iterations"] == 0


def test_chandrupatla_alias_resolves_to_canonical():
    # The `chandrupatlas` spelling resolves to the canonical engine name in the reply.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the `chandrupatlas` alias\n"
        "the reply reports the canonical 'chandrupatla'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatlas")
    assert payload["error"] is None
    assert payload["algorithm"] == "chandrupatla"


def test_chandrupatla_solves_keplers_equation():
    # The transcendental-root benchmark via Chandrupatla: smooth and monotone in [0, pi],
    # so the criterion admits the interpolation and the eccentric anomaly falls out fast.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via Chandrupatla\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="chandrupatla")
    assert payload["error"] is None
    assert payload["algorithm"] == "chandrupatla"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


# --- Secant: the chord through the last two points (33.3) ---------------------
# The fifth engine on the sign-change bracket and the plainest of them: no interpolation,
# no criterion — just the zero of the straight line through the two latest iterates,
# x2 = x1 - f1*(x1 - x0)/(f1 - f0). Newton's step with a finite difference in place of the
# derivative, so one evaluation per step and order ~1.618. Textbook secant can wander out
# of the interval; here lo/hi track the sign change and any escaping step is replaced by a
# bisection of that safeguard bracket, which costs nothing when the chord behaves.


def test_secant_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the secant method (endpoints straddle\n"
        "zero: f(0) = -2, f(2) = 2); expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="secant")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "secant"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    assert payload["iterations"] > 0
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_secant_converges_in_few_iterations():
    # Order ~1.618 on a smooth simple root, so a handful of steps rather than the ~35
    # bisection's halving would take.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via secant; converges in a few steps\n"
        "(the chord through the last two points is superlinear on a smooth root)",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="secant")
    assert payload["error"] is None
    assert payload["iterations"] <= 10


def test_secant_beats_chandrupatla_on_a_simple_root():
    # The trade 33.3 buys, one side of it: on a smooth simple root the bare chord needs
    # FEWER evaluations than the interpolators, because it spends none of them on the
    # third point their fit requires.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] — a smooth SIMPLE root, where the plain\n"
        "chord is the leanest step of the family: fewer evaluations than Chandrupatla",
        "x**2 - 2",
    )
    secant = _solve(program, variable="x", lower=0, upper=2, algorithm="secant")
    chand = _solve(program, variable="x", lower=0, upper=2, algorithm="chandrupatla")
    assert secant["error"] is None and chand["error"] is None
    assert _num(secant["solution"]) == pytest.approx(2**0.5, abs=1e-9)
    assert secant["iterations"] < chand["iterations"]


def test_secant_loses_to_chandrupatla_on_a_repeated_root():
    # The other side of the trade, pinned so the docstring's warning cannot rot: on the
    # triple root of x**3 the chord's slope collapses with the function and convergence
    # drops to linear, while Chandrupatla detects the case and runs at bisection's rate.
    program = _annotated(
        "find-root of x**3 over [-1, 2] — a TRIPLE root at x = 0. The chord's slope\n"
        "vanishes with the function, so secant degrades to linear convergence and needs\n"
        "far more steps than Chandrupatla, which bisects instead",
        "x**3",
    )
    secant = _solve(program, variable="x", lower=-1, upper=2, algorithm="secant")
    chand = _solve(program, variable="x", lower=-1, upper=2, algorithm="chandrupatla")
    assert secant["error"] is None and chand["error"] is None
    # Slower, but never less accurate — both still land the root exactly.
    assert _num(secant["solution"]) == 0.0
    assert _num(chand["solution"]) == 0.0
    assert secant["iterations"] > chand["iterations"]


def test_secant_safeguard_keeps_the_chord_inside_the_bracket():
    # Why the fence exists. ln(x) + 5 has its root at e**-5 ~ 0.006738, hard against the
    # log's vertical asymptote: the chord across that cell aims WELL left of the bracket
    # (unfenced, the second step lands at x ~ -0.22, where the log has no real value and
    # the search dies with no solution). The safeguard replaces exactly those steps with a
    # bisection, so the root is still found.
    program = _annotated(
        "find-root of ln(x) + 5 over [0.0001, 10] via secant; the root e**-5 ~ 0.006738\n"
        "sits against the log's asymptote, so the chord repeatedly aims outside the\n"
        "bracket — the bisection safeguard catches every such step",
        "ln(x) + 5",
    )
    payload = _solve(program, variable="x", lower=0.0001, upper=10, algorithm="secant")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(math.exp(-5), rel=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_secant_scans_when_endpoints_share_a_sign():
    # Like the rest of the family, secant scans for the sign change, so same-sign
    # endpoints (here f(-2) = f(2) = 2) are no obstacle: it brackets the leftmost root.
    program = _annotated(
        "find-root of x**2 - 2 over [-2, 2] via secant; both endpoints are +2,\n"
        "so the scan hunts the sign change. Leftmost root is x = -sqrt(2) ~ -1.41421",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="secant")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(-(2**0.5), abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_secant_finds_an_exact_root_on_the_fixed_point_grid():
    # The fixed-point grid polish applies here exactly as to the rest of the family: the
    # refined estimate lands within one grid step of x = 1.5, and the neighbour probes pin
    # it to an EXACT zero on the grid.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via secant in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=3,
        mode="fixed-point",
        floor=1,
        algorithm="secant",
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_secant_reports_no_sign_change():
    # x**2 + 1 never crosses zero, so the scan finds no sign-changing cell: the distinct
    # "no sign change" error, naming the engine and pointing at the alternatives.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via secant\n"
        "never crosses zero, so there is no sign change to bracket",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="secant")
    assert payload["solution"] is None and payload["value"] is None
    assert "No sign change" in payload["error"]
    assert "secant" in payload["error"]
    assert "find-minimum" in payload["error"]


def test_secant_rejects_a_non_root_objective():
    # It needs a straddle to fence the chord, which only locates a ROOT; find-minimum is
    # refused with a pointer to the engines that do minimise.
    program = _annotated(
        "secant asked to find-minimum — refused, it only finds roots\n"
        "(an extremum has no sign change to bracket)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="secant",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "brent-parabolic" in payload["error"]


def test_secant_rejects_the_multiple_form():
    # Single-variable like the other 1-D engines: the `variables` form needs Nelder-Mead.
    program = _annotated(
        "secant asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="secant")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_secant_skips_domain_failures_in_the_scan():
    # The left half raises a domain error (sqrt of a negative); those cells carry no
    # signed value and are skipped, yet the crossing at x = 1 is still bracketed.
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] via secant (the bracket dips below 0)\n"
        "the negative side has no real value and is skipped; expect x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4, algorithm="secant")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_secant_unset_constant_surfaces_as_an_eval_error():
    # A structural failure (a constant the program never sets) fails everywhere and must
    # surface as a line-tagged eval error, not be mistaken for a domain gap to skip.
    program = _annotated(
        "find-root of a * x - 1 over [0, 2] via secant; `a` is never set,\n"
        "so it surfaces as an eval error, not a region to skip",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="secant")
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


def test_secant_root_snaps_to_an_integer():
    # The float snap polish applies here too: a crossing a few ULPs off a clean integer
    # is re-snapped onto it (exact `==`, not approx).
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] via secant in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, algorithm="secant")
    assert payload["error"] is None
    assert payload["algorithm"] == "secant"
    assert _num(payload["solution"]) == 5.0


def test_secant_takes_a_root_sitting_on_a_scan_node():
    # x = 1 is exactly a scan node of [0, 2], so the cell before it straddles with an
    # endpoint value of precisely zero — which IS the moving end here. The f1 == 0 test
    # sees it before any chord is drawn, so the search ends without a single step.
    program = _annotated(
        "find-root of x - 1 over [0, 2] via secant; x = 1 lands exactly on a scan node,\n"
        "so the zero residual is already in hand and no chord is ever drawn",
        "x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="secant")
    assert payload["error"] is None
    assert _num(payload["solution"]) == 1.0
    assert _num(payload["value"]) == 0.0
    assert payload["iterations"] == 0


def test_secant_handles_a_zero_width_bracket():
    # The truly degenerate case: the root IS the first scan node, so the scan hands the
    # loop a == b — a zero-width bracket, which the width test recognises immediately
    # rather than dividing by a chord through one repeated point.
    program = _annotated(
        "find-root of x over [0, 2] via secant; the root x = 0 is the FIRST scan node,\n"
        "so the bracket handed to the loop has zero width — it must stop, not divide",
        "x",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="secant")
    assert payload["error"] is None
    assert _num(payload["solution"]) == 0.0
    assert _num(payload["value"]) == 0.0
    assert payload["iterations"] == 0


def test_secant_alias_resolves_to_canonical():
    # The `chord` spelling — the other textbook name for the step — resolves to the
    # canonical engine name in the reply.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the `chord` alias\n"
        "the reply reports the canonical 'secant'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="chord")
    assert payload["error"] is None
    assert payload["algorithm"] == "secant"


def test_secant_solves_keplers_equation():
    # The transcendental-root benchmark via secant: smooth and monotone in [0, pi], so
    # the chord tracks the curve closely and the eccentric anomaly falls out in a few steps.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via secant\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="secant")
    assert payload["error"] is None
    assert payload["algorithm"] == "secant"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


# --- Newton-Raphson: follow the tangent to zero (33.4) ------------------------
# The one root engine that brackets NOTHING. From a seed — the point of least |expr| in a
# coarse scan of the caller's interval — it repeatedly steps to where the tangent crosses
# zero, x - f(x)/f'(x), with f' differenced from the same five-point stencil the language's
# `diff` uses (40.17). Quadratic on a smooth simple root (a handful of steps), linear on a
# repeated one, and — because it never asks for a sign change — the only engine here that
# finds a root which merely TOUCHES zero. The bracket is a fence: every step is clamped
# back inside it, and a step that would leave stops the iteration.


def test_newton_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Newton-Raphson (no straddle needed:\n"
        "it follows the slope from a seed); expect x = sqrt(2) ~ 1.41421, expression ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-raphson")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "newton-raphson"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_newton_converges_in_few_iterations():
    # Quadratic convergence on a smooth simple root: each step roughly doubles the correct
    # digits, so a handful of steps against bisection's ~35 halvings.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Newton-Raphson; the tangent step\n"
        "doubles the correct digits each time, so it lands in a few steps",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-raphson")
    assert payload["error"] is None
    assert payload["iterations"] <= 6


def test_newton_beats_bisection_on_a_smooth_root():
    # The trade 33.4 buys, in steps: far fewer of them than the robust baseline. (Each is
    # dearer — five program evaluations against bisection's one — which is why the
    # docstring puts the comparison in iterations, not in work.)
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] — Newton's tangent step against bisection's\n"
        "halving: quadratic convergence needs a fraction of the iterations",
        "x**2 - 2",
    )
    newton = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-raphson")
    bisect = _solve(program, variable="x", lower=0, upper=2, algorithm="bisection")
    assert newton["error"] is None and bisect["error"] is None
    assert _num(newton["solution"]) == pytest.approx(2**0.5, abs=1e-9)
    assert newton["iterations"] < bisect["iterations"]


def test_newton_finds_a_root_the_bracketers_refuse():
    # WHY this engine earns its place. (x - pi)**2 only TOUCHES zero at x = pi — it never
    # changes sign — so every sign-change engine reports "no sign change" and refuses.
    # Newton needs no straddle: it follows the slope down to the root (linearly, since f
    # and f' vanish together there) and lands it. The root is irrational on purpose: at a
    # CLEAN touch root the float snap polish would hand bisection the answer anyway
    # (rounding its closest scan point onto the root), hiding the difference.
    program = _annotated(
        "find-root of (x - pi)**2 over [0, 5]: a DOUBLE root at x = pi, which only touches\n"
        "zero without crossing. Newton follows the slope to it; bisection has no sign\n"
        "change to bracket and refuses",
        "(x - pi)**2",
    )
    newton = _solve(program, variable="x", lower=0, upper=5, algorithm="newton-raphson")
    bisect = _solve(program, variable="x", lower=0, upper=5, algorithm="bisection")
    assert newton["error"] is None
    assert _num(newton["solution"]) == pytest.approx(math.pi, abs=1e-6)
    assert abs(_num(newton["value"])) < 1e-6
    assert bisect["solution"] is None
    assert "No sign change" in bisect["error"]


def test_newton_slows_to_linear_on_a_repeated_root():
    # The other side of that: quadratic convergence is a SIMPLE-root property. On the
    # triple root of x**3 the tangent step shrinks the error by a constant factor instead
    # of squaring it, so the same accuracy costs an order of magnitude more steps than the
    # simple root above — pinned so the docstring's warning cannot rot.
    simple = _solve(
        _annotated("find-root of x**2 - 2 over [0, 2] — a SIMPLE root", "x**2 - 2"),
        variable="x",
        lower=0,
        upper=2,
        algorithm="newton-raphson",
    )
    repeated = _solve(
        _annotated(
            "find-root of x**3 over [-1, 2] — a TRIPLE root at x = 0, where f and f'\n"
            "vanish together and Newton drops to linear convergence",
            "x**3",
        ),
        variable="x",
        lower=-1,
        upper=2,
        algorithm="newton-raphson",
    )
    assert simple["error"] is None and repeated["error"] is None
    # Slower, but never less accurate — the root still comes out exactly.
    assert _num(repeated["solution"]) == 0.0
    assert repeated["iterations"] > 10 * simple["iterations"]


def test_newton_fence_keeps_the_step_inside_the_bracket():
    # Why the fence exists. ln(x) + 5 has its root at e**-5 ~ 0.006738, hard against the
    # log's vertical asymptote: an unfenced tangent from out in the flat part of the curve
    # aims far to the left, where the log has no real value at all. Every step is clamped
    # back into [lower, upper], so the iteration stays where the expression is defined.
    program = _annotated(
        "find-root of ln(x) + 5 over [0.0001, 10] via Newton-Raphson; the root e**-5\n"
        "~ 0.006738 sits against the log's asymptote, so the tangent repeatedly aims\n"
        "outside the bracket — the fence clamps every such step back inside",
        "ln(x) + 5",
    )
    payload = _solve(program, variable="x", lower=0.0001, upper=10, algorithm="newton-raphson")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(math.exp(-5), rel=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_newton_keeps_its_answer_inside_the_bracket():
    # The derivative stencil probes x +- h and x +- 2h, which near an endpoint fall OUTSIDE
    # the caller's bracket; being real evaluations they used to feed best-tracking, so a
    # probe landing on a root just past the endpoint was RETURNED as the answer. x + 0.00001
    # has its only root at -1e-5, just below the bracket, and a stencil probe from the seed
    # at x = 0 lands on it exactly — the case that exposed the leak. The answer must stay in
    # [0, 5], so the engine reports no in-bracket solution, exactly as the bracketers do.
    program = _annotated(
        "find-root of x + 0.00001 over [0, 5] via Newton-Raphson — the only root is at\n"
        "-1e-5, OUTSIDE the bracket; a stencil probe must not leak it back as the answer",
        "x + 0.00001",
    )
    payload = _solve(program, variable="x", lower=0, upper=5, algorithm="newton-raphson")
    assert payload["solution"] is None and payload["value"] is None
    assert "does not reach zero" in payload["error"]
    assert "[0.0, 5.0]" in payload["error"]  # the interval reported, never a point outside it


def test_newton_still_reaches_a_root_within_a_stencil_step_of_the_endpoint():
    # The other side of that fix: gating the RECORD (not the evaluation) means an in-bracket
    # root very close to an endpoint is still found — the iteration point stays fenced and
    # converges onto it, where a cruder "never probe past the endpoint" rule would give up.
    # x - 0.000001 has its root at 1e-6, just inside the lower endpoint.
    program = _annotated(
        "find-root of x - 0.000001 over [0, 5] via Newton-Raphson — root at 1e-6, a hair\n"
        "INSIDE the lower endpoint; it must still be found, not abandoned near the edge",
        "x - 0.000001",
    )
    payload = _solve(program, variable="x", lower=0, upper=5, algorithm="newton-raphson")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1e-6, abs=1e-9)
    assert _num(payload["solution"]) >= 0.0  # inside the bracket, never below `lower`


def test_newton_finds_an_exact_root_on_the_fixed_point_grid():
    # The fixed-point grid polish applies here exactly as to the bracketed family: the
    # estimate lands within one grid step of x = 1.5 and the neighbour probes pin it to an
    # EXACT zero on the grid.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via Newton-Raphson in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=3,
        mode="fixed-point",
        floor=1,
        algorithm="newton-raphson",
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_newton_reports_a_flat_tangent():
    # x**2 + 1 never reaches zero, and its least |expr| — where the seed lands — is the
    # vertex at x = 0, whose tangent is horizontal: there is no step to take. That is a
    # named outcome, distinct from the bracketers' "no sign change", and it points at the
    # engines to try instead.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via Newton-Raphson; it never reaches zero,\n"
        "and the seed lands on the flat vertex — a horizontal tangent, nowhere to step",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="newton-raphson")
    assert payload["solution"] is None and payload["value"] is None
    assert "No solution" in payload["error"]
    assert "tangent went flat" in payload["error"]
    assert "bisection" in payload["error"]


def test_newton_rejects_a_non_root_objective():
    # The tangent construction locates a ZERO, not an extremum; find-minimum is refused
    # with a pointer to the engines that do minimise.
    program = _annotated(
        "Newton-Raphson asked to find-minimum — refused, it only finds roots\n"
        "(the tangent step crosses zero; an extremum is another question)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="newton-raphson",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "brent-parabolic" in payload["error"]


def test_newton_rejects_the_multiple_form():
    # Single-variable like the other 1-D engines: the `variables` form needs Nelder-Mead.
    program = _annotated(
        "Newton-Raphson asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="newton-raphson")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_newton_skips_domain_failures_in_the_seed_scan():
    # The left half raises a domain error (sqrt of a negative); those scan points carry no
    # value and cannot be seeds, so the seed comes from where the expression IS defined and
    # the crossing at x = 1 is still found.
    program = _annotated(
        "find-root of sqrt(x) - 1 over [-1, 4] via Newton-Raphson (the bracket dips below\n"
        "0); the negative side has no real value and is skipped, x = 1 is still found",
        "sqrt(x) - 1",
    )
    payload = _solve(program, variable="x", lower=-1, upper=4, algorithm="newton-raphson")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.0, abs=1e-6)


def test_newton_unset_constant_surfaces_as_an_eval_error():
    # A structural failure (a constant the program never sets) fails everywhere and must
    # surface as a line-tagged eval error, not be mistaken for a domain gap to skip.
    program = _annotated(
        "find-root of a * x - 1 over [0, 2] via Newton-Raphson; `a` is never set,\n"
        "so it surfaces as an eval error, not a region to skip",
        "a * x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-raphson")
    assert payload["solution"] is None
    assert "undefined variable: a" in payload["error"]


def test_newton_root_snaps_to_an_integer():
    # The float snap polish applies here too: an answer a few ULPs off a clean integer is
    # re-snapped onto it (exact `==`, not approx).
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] via Newton-Raphson in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, algorithm="newton-raphson")
    assert payload["error"] is None
    assert payload["algorithm"] == "newton-raphson"
    assert _num(payload["solution"]) == 5.0


def test_newton_takes_a_root_sitting_on_a_seed_scan_node():
    # x = 1 is exactly a scan node of [0, 2], so the seed already HAS a zero residual: the
    # f(x) == 0 test sees it before any tangent is drawn and the search ends in no steps.
    program = _annotated(
        "find-root of x - 1 over [0, 2] via Newton-Raphson; x = 1 lands exactly on a seed\n"
        "scan node, so the zero residual is in hand and no tangent is ever drawn",
        "x - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-raphson")
    assert payload["error"] is None
    assert _num(payload["solution"]) == 1.0
    assert _num(payload["value"]) == 0.0
    assert payload["iterations"] == 0


def test_newton_alias_resolves_to_canonical():
    # Bare `newton` — the spelling a caller reaches for first — resolves to the canonical
    # engine name in the reply.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the bare `newton` alias\n"
        "the reply reports the canonical 'newton-raphson'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="newton")
    assert payload["error"] is None
    assert payload["algorithm"] == "newton-raphson"


def test_newton_solves_keplers_equation():
    # The transcendental-root benchmark via Newton: smooth and monotone in [0, pi], the
    # case the tangent step was made for — the eccentric anomaly falls out in a few steps.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via Newton-Raphson\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="newton-raphson")
    assert payload["error"] is None
    assert payload["algorithm"] == "newton-raphson"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


# --- Halley: Newton's step with the curvature in it (33.8) --------------------
# The second tangent engine, sharing every part of Newton's harness but the step: where
# Newton fits the curve by its tangent LINE, Halley fits a hyperbola matching the value,
# the slope and the curvature, x - 2ff'/(2f'**2 - ff''). One order up (the error is cubed
# each step, not squared) for exactly the same five program evaluations, because both
# derivatives come out of the SAME five-point stencil. The curvature term is trusted only
# while it keeps the denominator at half of 2f'**2 or better; outside that it takes
# Newton's step, so Halley is never much worse than the engine it extends.


def test_halley_finds_a_root():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via Halley (Newton's tangent step with the\n"
        "curvature added); expect x = sqrt(2) ~ 1.41421, where the expression is ~0",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="halley")
    assert payload["error"] is None
    assert payload["objective"] == "find-root"
    assert payload["algorithm"] == "halley"
    assert _num(payload["solution"]) == pytest.approx(2**0.5, abs=1e-6)
    assert abs(_num(payload["value"])) < 1e-6
    # The single-unknown convenience: the scalar fields echo the one solutions entry.
    assert payload["variable"] == "x"
    assert [entry["variable"] for entry in payload["solutions"]] == ["x"]


def test_halley_beats_newton_on_a_smooth_root():
    # The trade 33.8 buys: cubic against quadratic convergence, for the same number of
    # program evaluations per step (the curvature is differenced from samples Newton
    # already pays for). So on a smooth root it is strictly the better step.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] — Halley's cubic step against Newton's\n"
        "quadratic one: the same five evaluations per step, fewer steps",
        "x**2 - 2",
    )
    halley = _solve(program, variable="x", lower=0, upper=2, algorithm="halley")
    newton = _solve(program, variable="x", lower=0, upper=2, algorithm="newton-raphson")
    assert halley["error"] is None and newton["error"] is None
    assert _num(halley["solution"]) == pytest.approx(2**0.5, abs=1e-9)
    assert halley["iterations"] < newton["iterations"]


def test_halley_matches_newton_where_the_curve_is_straight():
    # The other end of the same statement: Halley's step is Newton's scaled by the
    # curvature factor 1/(1 - f*f''/(2*f'**2)), so on a LINEAR expression (f'' = 0) the
    # factor is exactly 1 and the two engines are the same iteration, step for step.
    program = _annotated(
        "find-root of 3*x - 7 over [0, 8]: a straight line, so there is no curvature for\n"
        "Halley to add — it must take Newton's step exactly, and land the same 7/3",
        "3*x - 7",
    )
    halley = _solve(program, variable="x", lower=0, upper=8, algorithm="halley")
    newton = _solve(program, variable="x", lower=0, upper=8, algorithm="newton-raphson")
    assert halley["error"] is None and newton["error"] is None
    assert _num(halley["solution"]) == _num(newton["solution"]) == pytest.approx(7 / 3, abs=1e-9)
    assert halley["iterations"] == newton["iterations"]


def test_halley_falls_back_to_newtons_step_at_an_asymptote():
    # REGRESSION on the safeguard's SIGN test. ln(x) + 5 has its root at e**-5 ~ 0.006738,
    # hard against the log's vertical asymptote, where the curvature is enormous: f*f''
    # overwhelms 2*f'**2 and turns the Halley denominator NEGATIVE, which would reverse the
    # step and walk the iteration out of the bracket (an unguarded version fails here while
    # Newton succeeds). Requiring the denominator to keep at least half of 2*f'**2 — a
    # signed test, not a magnitude one — falls back to the Newton step exactly there.
    program = _annotated(
        "find-root of ln(x) + 5 over [0.0001, 10] via Halley; at the log's asymptote the\n"
        "curvature term flips the step's sign, so the safeguard falls back to Newton's\n"
        "step and the root e**-5 ~ 0.006738 is still found",
        "ln(x) + 5",
    )
    payload = _solve(program, variable="x", lower=0.0001, upper=10, algorithm="halley")
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(math.exp(-5), rel=1e-6)
    assert abs(_num(payload["value"])) < 1e-6


def test_halley_finds_a_root_the_bracketers_refuse():
    # The tangent family's shared payoff, on the second engine: (x - pi)**2 only TOUCHES
    # zero, so no sign change exists to bracket, but a derivative step walks straight down
    # to it. (Irrational root on purpose — see the Newton twin of this test.)
    program = _annotated(
        "find-root of (x - pi)**2 over [0, 5] via Halley: a DOUBLE root at x = pi that\n"
        "only touches zero. No sign change exists, and the bracketed engines refuse it",
        "(x - pi)**2",
    )
    halley = _solve(program, variable="x", lower=0, upper=5, algorithm="halley")
    brent = _solve(program, variable="x", lower=0, upper=5, algorithm="brent-dekker")
    assert halley["error"] is None
    assert _num(halley["solution"]) == pytest.approx(math.pi, abs=1e-6)
    assert abs(_num(halley["value"])) < 1e-6
    assert brent["solution"] is None
    assert "No sign change" in brent["error"]


def test_halley_finds_an_exact_root_on_the_fixed_point_grid():
    # The grid polish applies to this engine as to every other single-unknown one.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 3] via Halley in fixed-point (scale 1)\n"
        "expect x = 1.5 exactly: it lands on the grid, an EXACT zero (grid polish)",
        "2*x - 3",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=3,
        mode="fixed-point",
        floor=1,
        algorithm="halley",
    )
    assert payload["error"] is None
    assert payload["solution"].split(" (")[0] == "1.5"
    assert _num(payload["value"]) == 0.0
    assert payload["exact"] is True


def test_halley_keeps_its_answer_inside_the_bracket():
    # The same stencil-leak fix as its Newton twin, on the second tangent engine: an
    # out-of-bracket derivative probe must never be returned. x + 0.00001's only root is at
    # -1e-5, below the bracket, so the answer must stay inside [0, 5] and report no
    # in-bracket solution.
    program = _annotated(
        "find-root of x + 0.00001 over [0, 5] via Halley — the only root is at -1e-5,\n"
        "OUTSIDE the bracket; a stencil probe must not leak it back as the answer",
        "x + 0.00001",
    )
    payload = _solve(program, variable="x", lower=0, upper=5, algorithm="halley")
    assert payload["solution"] is None and payload["value"] is None
    assert "does not reach zero" in payload["error"]
    assert "[0.0, 5.0]" in payload["error"]


def test_halley_reports_a_flat_tangent():
    # The shared stop of the tangent family: x**2 + 1 never reaches zero and its least
    # |expr| is the vertex at x = 0, where the slope is zero. Halley's own denominator is
    # not what fails — a zero f' leaves no step for either engine — so the same named
    # outcome is reported.
    program = _annotated(
        "find-root of x**2 + 1 over [-2, 2] via Halley; it never reaches zero, and the\n"
        "seed lands on the flat vertex — a horizontal tangent, nowhere to step",
        "x**2 + 1",
    )
    payload = _solve(program, variable="x", lower=-2, upper=2, algorithm="halley")
    assert payload["solution"] is None and payload["value"] is None
    assert "No solution" in payload["error"]
    assert "tangent went flat" in payload["error"]
    assert "bisection" in payload["error"]


def test_halley_rejects_a_non_root_objective():
    # A derivative step locates a ZERO, not an extremum — refused, like its Newton sibling.
    program = _annotated(
        "Halley asked to find-minimum — refused, it only finds roots\n"
        "(the step crosses zero; an extremum is another question)",
        "(x - 3)**2",
    )
    payload = _solve(
        program,
        variable="x",
        lower=0,
        upper=5,
        objective="find-minimum",
        algorithm="halley",
    )
    assert payload["solution"] is None
    assert "only finds roots" in payload["error"]
    assert "brent-parabolic" in payload["error"]


def test_halley_rejects_the_multiple_form():
    # Single-variable like the other 1-D engines: the `variables` form needs Nelder-Mead.
    program = _annotated(
        "Halley asked to solve TWO unknowns — refused, it is single-variable\n"
        "(the variables form needs algorithm='nelder-mead')",
        "x + y",
    )
    payload = _solve(program, variables={"x": [0, 1], "y": [0, 1]}, algorithm="halley")
    assert payload["solution"] is None and payload["solutions"] is None
    assert "single variable" in payload["error"] and "nelder-mead" in payload["error"]


def test_halley_alias_resolves_to_canonical():
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] via the `halleys-method` alias\n"
        "the reply reports the canonical 'halley'",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, algorithm="halleys-method")
    assert payload["error"] is None
    assert payload["algorithm"] == "halley"


def test_halley_solves_keplers_equation():
    # The transcendental-root benchmark on the fastest engine here: smooth and monotone in
    # [0, pi], so the cubic step lands the eccentric anomaly in a couple of iterations.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi] via Halley\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi, algorithm="halley")
    assert payload["error"] is None
    assert payload["algorithm"] == "halley"
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


# --- REGRESSION: floating-point answers snap onto their clean value -----------
# The drift these tests once reproduced: on a problem whose true answer is a whole
# number, the floating-point search settled a few ULPs away — 4.999999999999984 for
# 5, 3.9999999717 for 4 — because the search stops within the bracket-width tolerance
# (_FLOAT_X_TOL) and the fixed-point / rational grid polish that would re-snap it is
# skipped in float mode (float has no grid). The fix is its float counterpart,
# solver._float_snap_polish: after convergence it re-probes the CLEAN roundings of the
# best point (0..6 decimals) through the same evaluate_objective best-tracking, which
# adopts one only when it is no worse. So a clean root / optimum snaps onto its exact
# value, while an irrational answer (whose rounding is strictly worse) is left alone —
# the objective itself is the discriminator (the "irrational stays put" tests below
# guard that). The assertions use exact `==` (no pytest.approx) on purpose: approx
# would paper over the very drift these tests exist to catch. Both objective kinds
# (root-finding and optimisation) are covered for each engine; Brent's parabola pins a
# smooth extremum exactly, so it is exercised on the kinked |expr| of a root.


def test_golden_section_root_snaps_to_an_integer():
    program = _annotated(
        "find-root of 2*x - 10 over [0, 8] in floating-point\n"
        "the only root is x = 5 exactly; snap polish returns a clean 5, not 4.999999999999984",
        "2*x - 10",
    )
    payload = _solve(program, variable="x", lower=0, upper=8)
    assert payload["error"] is None
    assert payload["algorithm"] == "golden-section-search"
    assert _num(payload["solution"]) == 5.0


def test_golden_section_maximum_snaps_to_an_integer():
    program = _annotated(
        "find-maximum of 10 - (x - 4)**2 over [0, 8] in floating-point\n"
        "the peak is at x = 4 exactly; snap polish returns a clean 4, not 3.9999999717159223",
        "10 - (x - 4)**2",
    )
    payload = _solve(program, variable="x", lower=0, upper=8, objective="find-maximum")
    assert payload["error"] is None
    assert payload["algorithm"] == "golden-section-search"
    assert _num(payload["solution"]) == 4.0


def test_brent_parabolic_root_snaps_to_an_integer():
    program = _annotated(
        "find-root of x**2 - 25 over [0, 10] in floating-point via Brent\n"
        "the root in range is x = 5 exactly; snap polish returns a clean 5, not 5.000000000000054",
        "x**2 - 25",
    )
    payload = _solve(program, variable="x", lower=0, upper=10, algorithm="brent-parabolic")
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-parabolic"
    assert _num(payload["solution"]) == 5.0


def test_ternary_root_snaps_to_an_integer():
    program = _annotated(
        "find-root of x**2 - 25 over [0, 10] in floating-point via ternary search\n"
        "the root in range is x = 5 exactly; snap polish returns a clean 5, not 4.999999999999984",
        "x**2 - 25",
    )
    payload = _solve(program, variable="x", lower=0, upper=10, algorithm="ternary-search")
    assert payload["error"] is None
    assert payload["algorithm"] == "ternary-search"
    assert _num(payload["solution"]) == 5.0


def test_nelder_mead_root_snaps_to_an_integer():
    program = _annotated(
        "find-root of x**2 - 49 over [0, 12] in floating-point via Nelder-Mead\n"
        "the root in range is x = 7 exactly; snap polish returns a clean 7, not 7.000000000000183",
        "x**2 - 49",
    )
    payload = _solve(program, variables={"x": [0, 12]}, algorithm="nelder-mead")
    assert payload["error"] is None
    assert payload["algorithm"] == "nelder-mead"
    assert _num(payload["solution"]) == 7.0


def test_nelder_mead_two_variable_minimum_snaps_to_integers():
    program = _annotated(
        "find-minimum of (x - 3)**2 + (y - 7)**2 over x in [0, 6], y in [0, 12]\n"
        "the single minimum is at (3, 7) exactly; snap polish returns clean 3 and 7",
        "(x - 3)**2 + (y - 7)**2",
    )
    payload = _solve(
        program,
        variables={"x": [0, 6], "y": [0, 12]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == 3.0
    assert found["y"] == 7.0


def test_root_snaps_to_a_clean_half_integer():
    # The ladder is decimals, not just integers: a root at 1.5 that drifts to ~1.4999998
    # snaps at one decimal place. Rounding to the NEAREST integer (1 or 2) is rejected —
    # it is not a root — so only the genuinely-clean 1.5 is adopted.
    program = _annotated(
        "find-root of 2*x - 3 over [0, 4] in floating-point\nthe root is x = 1.5 exactly",
        "2*x - 3",
    )
    payload = _solve(program, variable="x", lower=0, upper=4)
    assert payload["error"] is None
    assert _num(payload["solution"]) == 1.5


def test_irrational_root_is_not_snapped():
    # The guard: sqrt(2) is irrational, so no clean rounding is a root — every probe is
    # strictly worse than the converged point and best-tracking keeps the latter. The
    # answer stays ~1.41421 and is NOT snapped to 1, 1.4, or any short decimal.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] in floating-point\n"
        "the root is sqrt(2) ~ 1.41421 — irrational, so it must stay put, not snap",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2)
    assert payload["error"] is None
    found = _num(payload["solution"])
    assert found == pytest.approx(2**0.5, abs=1e-6)
    assert found != 1.0 and found != 2.0 and found != 1.4


def test_irrational_optimum_is_not_snapped():
    # The same guard for an extremum: (x**2 - 2)**2 bottoms out at x = sqrt(2); a rounded
    # x gives a strictly larger value, so the minimiser is left at ~1.41421, not snapped.
    program = _annotated(
        "find-minimum of (x**2 - 2)**2 over [0, 2] in floating-point\n"
        "the minimiser is sqrt(2) ~ 1.41421 — irrational, so it must not snap to 1 or 1.4",
        "(x**2 - 2)**2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, objective="find-minimum")
    assert payload["error"] is None
    found = _num(payload["solution"])
    assert found == pytest.approx(2**0.5, abs=1e-5)
    assert found != 1.0 and found != 1.4


# --- BENCHMARKS: classic root-finding and optimisation test problems ----------
# Named problems from the numerical-methods literature, driven over the same tool
# seam. The single-variable transcendental roots (Kepler, Dottie, Omega) exercise
# the 1-D engines on smooth non-polynomial residuals; the 2-D surfaces (Rosenbrock,
# Himmelblau, Booth, Beale) exercise Nelder-Mead on the optimisation shapes it is
# usually benchmarked against — a curved valley, a multimodal field, a convex bowl,
# and a sharp asymmetric basin. Boxes are placed so the LOCAL, derivative-free
# engines converge to the intended (global) optimum; reference values are abs-1e-3
# or tighter, with the residual checked to confirm a genuine root / floor.


def test_keplers_equation_root():
    # Kepler's equation E - e*sin(E) = M (orbital mechanics): given eccentricity
    # e = 0.8 and mean anomaly M = 1 rad, find the eccentric anomaly E. Smooth and
    # monotone in [0, pi], so golden-section drives |expr| straight to zero.
    program = _annotated(
        "find-root of Kepler's equation E - 0.8*sin(E) - 1 over [0, pi]\n"
        "eccentricity 0.8, mean anomaly 1 rad; expect eccentric anomaly E ~ 1.782191",
        "E - 0.8*sin(E) - 1",
    )
    payload = _solve(program, variable="E", lower=0, upper=math.pi)
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(1.7821913289379006, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


def test_dottie_number_via_brent():
    # The Dottie number: the unique real fixed point of cosine, the root of
    # cos(x) - x. A smooth transcendental root for Brent's parabolic minimiser.
    program = _annotated(
        "find-root of cos(x) - x over [0, 1] via Brent (the Dottie number)\n"
        "the unique real fixed point of cosine; expect x ~ 0.739085",
        "cos(x) - x",
    )
    payload = _solve(program, variable="x", lower=0, upper=1, algorithm="brent-parabolic")
    assert payload["error"] is None
    assert payload["algorithm"] == "brent-parabolic"
    assert _num(payload["solution"]) == pytest.approx(0.7390851332151607, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


def test_omega_constant_root():
    # The omega constant Ω = W(1): the root of x*e^x - 1, where W is the Lambert W
    # function. Another well-known transcendental constant defined by a root.
    program = _annotated(
        "find-root of x*exp(x) - 1 over [0, 1] (the omega constant, W(1))\n"
        "expect x ~ 0.567143, where x*e^x = 1",
        "x*exp(x) - 1",
    )
    payload = _solve(program, variable="x", lower=0, upper=1)
    assert payload["error"] is None
    assert _num(payload["solution"]) == pytest.approx(0.5671432904097837, abs=1e-3)
    assert abs(_num(payload["value"])) < 1e-6


def test_rosenbrock_minimum():
    # Rosenbrock's banana function, the canonical optimisation benchmark: a long
    # curved valley with the global minimum at (1, 1), value 0. Hard for naive
    # methods, but Nelder-Mead tracks the valley floor down to the corner.
    program = _annotated(
        "find-minimum of Rosenbrock 100*(y - x**2)**2 + (1 - x)**2\n"
        "over x in [-2, 2], y in [-1, 3]; global minimum at (1, 1), value 0",
        "100*(y - x**2)**2 + (1 - x)**2",
    )
    payload = _solve(
        program,
        variables={"x": [-2, 2], "y": [-1, 3]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == pytest.approx(1.0, abs=1e-3)
    assert found["y"] == pytest.approx(1.0, abs=1e-3)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)


def test_himmelblau_minimum():
    # Himmelblau's function: a multimodal field with FOUR equal minima of value 0.
    # The local simplex converges to whichever basin its box brackets — here [0, 5]^2
    # holds the (3, 2) minimum, and that is the one found.
    program = _annotated(
        "find-minimum of Himmelblau (x**2 + y - 11)**2 + (x + y**2 - 7)**2\n"
        "over x, y in [0, 5]; four equal minima exist, this box holds (3, 2), value 0",
        "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
    )
    payload = _solve(
        program,
        variables={"x": [0, 5], "y": [0, 5]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == pytest.approx(3.0, abs=1e-3)
    assert found["y"] == pytest.approx(2.0, abs=1e-3)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)


def test_booth_minimum():
    # Booth's function: a smooth convex bowl (two linear forms squared) with a single
    # minimum at (1, 3), value 0 — equivalently the least-squares solution of the
    # linear system x + 2y = 7, 2x + y = 5.
    program = _annotated(
        "find-minimum of Booth (x + 2*y - 7)**2 + (2*x + y - 5)**2\n"
        "over x, y in [-5, 5]; single minimum at (1, 3), value 0",
        "(x + 2*y - 7)**2 + (2*x + y - 5)**2",
    )
    payload = _solve(
        program,
        variables={"x": [-5, 5], "y": [-5, 5]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == pytest.approx(1.0, abs=1e-3)
    assert found["y"] == pytest.approx(3.0, abs=1e-3)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)


def test_beale_minimum():
    # Beale's function: a sharp, asymmetric basin with the global minimum at
    # (3, 0.5), value 0. The cubic y-term makes it steep, a classic stress test for
    # derivative-free optimisers.
    program = _annotated(
        "find-minimum of Beale (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2\n"
        "+ (2.625 - x + x*y**3)**2 over x in [0, 4.5], y in [-1, 1]\n"
        "global minimum at (3, 0.5), value 0",
        "(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2",
    )
    payload = _solve(
        program,
        variables={"x": [0, 4.5], "y": [-1, 1]},
        objective="find-minimum",
        algorithm="nelder-mead",
    )
    assert payload["error"] is None
    found = {entry["variable"]: _num(entry["solution"]) for entry in payload["solutions"]}
    assert found["x"] == pytest.approx(3.0, abs=1e-3)
    assert found["y"] == pytest.approx(0.5, abs=1e-3)
    assert _num(payload["value"]) == pytest.approx(0.0, abs=1e-6)


# --- 43.3: auto-detecting an omitted `variable` -------------------------------
# The single-unknown form may omit `variable`; the solver infers it as the
# expression's sole free name (referenced but not assigned), needing only the
# lower+upper bracket. Detection is refused when it is not unique (zero or >1).


def test_autodetect_single_variable():
    program = _annotated(
        "find-root of 12*n - (450 + 3*n) with `variable` OMITTED; the sole free\n"
        "name n is auto-detected. 9n = 450 -> n = 50",
        "12*n - (450 + 3*n)",
    )
    payload = _solve(program, lower=0, upper=1000)
    assert payload["error"] is None
    assert payload["variable"] == "n"
    assert _num(payload["solution"]) == pytest.approx(50.0, abs=1e-6)


def test_autodetect_excludes_assigned_constants():
    # A program whose assignment lines set r and p leaves n as the only free name,
    # so auto-detect picks n even though three names appear: p*(1+r)**n = 2000.
    program = _annotated(
        "find-root of p*(1+r)**n - 2000 with r, p set by assignment lines and\n"
        "`variable` omitted; only the unassigned n is free. 1000*1.05**n = 2000",
        "r = 0.05\np = 1000\np * (1 + r)**n - 2000",
    )
    payload = _solve(program, lower=0, upper=100)
    assert payload["error"] is None
    assert payload["variable"] == "n"
    # log(2)/log(1.05) = 14.2067...
    assert _num(payload["solution"]) == pytest.approx(14.2067, abs=1e-3)


def test_autodetect_with_brent_and_minimum():
    # Detection is independent of engine and objective: a parabola with its sole
    # free name x, minimised via brent-parabolic, `variable` omitted. min at x = 2.
    program = _annotated(
        "find-minimum of (x - 2)**2 + 1 with `variable` omitted; x auto-detected,\n"
        "brent-parabolic engine. minimum at x = 2",
        "(x - 2)**2 + 1",
    )
    payload = _solve(
        program, lower=-5, upper=5, objective="find-minimum", algorithm="brent-parabolic"
    )
    assert payload["error"] is None
    assert payload["variable"] == "x"
    assert _num(payload["solution"]) == pytest.approx(2.0, abs=1e-3)


def test_autodetect_ambiguous_multiple_free_names():
    program = _annotated(
        "two free names a and b: auto-detect is ambiguous and must be refused,\n"
        "telling the caller to name `variable`",
        "a + b",
    )
    payload = _solve(program, lower=0, upper=1)
    assert payload["error"] is not None
    assert "auto-detect" in payload["error"]
    assert "'a'" in payload["error"] and "'b'" in payload["error"]


def test_autodetect_no_free_name():
    program = _annotated(
        "a constant expression has no free name; auto-detect must be refused",
        "2 + 2",
    )
    payload = _solve(program, lower=0, upper=1)
    assert payload["error"] is not None
    assert "no free variable" in payload["error"]


def test_autodetect_detected_but_no_bracket():
    # The unknown is inferred but the single form still needs a bracket; the error
    # names the detected variable so the caller knows what to bound.
    program = _annotated(
        "n is auto-detected but lower+upper are missing; the error names n",
        "12*n - 450",
    )
    payload = _solve(program)
    assert payload["error"] is not None
    assert "No search bracket" in payload["error"]
    assert "'n'" in payload["error"]


def test_explicit_variable_still_works():
    # Regression: passing `variable` explicitly is unchanged by auto-detect.
    program = _annotated(
        "explicit variable n is honoured as before. 9n = 450 -> n = 50",
        "12*n - (450 + 3*n)",
    )
    payload = _solve(program, variable="n", lower=0, upper=1000)
    assert payload["error"] is None
    assert payload["variable"] == "n"
    assert _num(payload["solution"]) == pytest.approx(50.0, abs=1e-6)


# --- error paths: rejections and unevaluable domains over the seam -------------
# These exercise the validate-and-refuse branches the success tests never reach.
# Each pins the EXACT message and the full null-shape reply (every data field None),
# the solver's _solver_error mirror of calculate's _error.

_SOLVER_DATA_FIELDS = (
    "variable",
    "solution",
    "solution_hex_dump",
    "solutions",
    "value",
    "value_hex_dump",
    "mode",
    "exact",
    "precision",
    "objective",
    "algorithm",
    "iterations",
)


def test_solver_rejects_complex_mode():
    # The solver is real-valued (it minimises |expr| / brackets a sign change, both of
    # which need an ordering complex lacks), so a complex-mode request refuses outright —
    # and the whole reply is the null shape, only `error` populated.
    payload = _solve("x**2 - 2", variable="x", lower=0, upper=2, mode="complex")
    assert payload["error"] == "the solver is real-valued; complex mode is not supported"
    assert all(payload[field] is None for field in _SOLVER_DATA_FIELDS)


def test_solver_reports_when_the_expression_is_unevaluable_across_the_bracket():
    # DISTINCT from "No solution" (a value was found but missed zero): here EVERY candidate
    # raised a domain error (sqrt of a negative across the whole [-4, -1]), so the search
    # never had a value to compare — a different, named failure.
    payload = _solve("sqrt(x) - 1", variable="x", lower=-4, upper=-1)
    assert payload["error"] == (
        "The expression could not be evaluated anywhere in [-4.0, -1.0] "
        "(every candidate for 'x' raised a domain error)."
    )


def test_nelder_mead_reports_when_the_box_is_entirely_unevaluable():
    # The multi-unknown twin of the above: the Nelder-Mead box message names the box.
    payload = _solve("sqrt(x) - 1", variables={"x": [-4, -1]}, algorithm="nelder-mead")
    assert payload["error"] == (
        "The expression could not be evaluated anywhere in the search box "
        "(x in [-4.0, -1.0]) (every candidate raised a domain error)."
    )


def test_solver_rejects_giving_both_unknown_forms():
    # variable+lower+upper (single) XOR variables (multiple) — supplying both is ambiguous
    # about which search the caller wants, so it refuses rather than guess.
    payload = _solve("x + y", variable="x", lower=0, upper=1, variables={"y": [0, 1]})
    assert payload["error"] == (
        "Give exactly one unknown form: variable + lower + upper (single), "
        "or variables (multiple); not both."
    )


def test_solver_rejects_an_empty_variables_map():
    payload = _solve("x", variables={}, algorithm="nelder-mead")
    assert payload["error"] == "No unknowns given: 'variables' is empty."


def test_solver_rejects_a_malformed_bracket_pair():
    # Each `variables` entry must be a [lower, upper] PAIR; a three-element list is malformed
    # and the error echoes the (floated) list it got.
    payload = _solve("x", variables={"x": [0, 1, 2]}, algorithm="nelder-mead")
    assert payload["error"] == "Bracket for 'x' must be a [lower, upper] pair, got [0.0, 1.0, 2.0]."
