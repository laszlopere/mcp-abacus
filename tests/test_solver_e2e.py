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
