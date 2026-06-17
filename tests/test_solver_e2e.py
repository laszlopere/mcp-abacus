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
