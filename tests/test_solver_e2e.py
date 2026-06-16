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
        arguments["lower"] = lower
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


def test_fixed_point_solver_requires_min_fixed_point_precision():
    # 39: fixed-point is the default mode, but a search at scale 0 floors the variable
    # to whole numbers and would miss a non-integer root, so the bare fixed-point call
    # is refused — naming the argument and a concrete value to pass. The request is
    # otherwise well-formed (x occurs, bracket non-empty), so it reaches this check.
    program = _annotated(
        "find-root of x**2 - 2 over [0, 2] in fixed-point with NO precision\n"
        "refused: the search would floor x to integers and miss sqrt(2)",
        "x**2 - 2",
    )
    payload = _solve(program, variable="x", lower=0, upper=2, mode="fixed-point")
    assert payload["solution"] is None and payload["value"] is None
    assert "min_fixed_point_precision is required in fixed-point mode" in payload["error"]
    assert "e.g. 9" in payload["error"]


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
