# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Special forms — call functions whose arguments are NOT all evaluated operands (TODO 40).

The third callable kind beside the operand-methods (_FUNCS) and the nullaries
(_NULLARY_FUNCS) in expr.nodes. A special form receives the run EvalContext and its
arguments' UNEVALUATED subtrees, because some arguments are an unevaluated sub-program
and a free variable NAME rather than values — the solver's shape, brought inside the
expression language. ``integral(expr, var, a, b)`` (40.18) is the first: it re-evaluates
``expr`` at many sample points with ``var`` rebound to each, the way solver.py drives an
expression while searching.

This is a SIBLING engine to solver.py: it imports expr.nodes (Node.evaluate, the AST) to
re-run the integrand, an edge value.py cannot carry — which is why these live here and not
as Value methods. expr.nodes imports THIS module only at call time (FuncCall._evaluate),
so the back-edge never exists at module load.

Everything stays in the ACTIVE mode: a definite integral is just +, *, /, and one
comparison, all of which Value already does mode-faithfully, so the quadrature accumulates
with Value arithmetic — no float shadow (unlike the solver, whose golden-section / parabola
math is intrinsically floating-point). The result is flagged inexact in EVERY mode (40.18),
because Simpson's rule only APPROXIMATES the true integral — not because of any conversion.
"""

from fractions import Fraction

from mcp_abacus.expr.nodes import EvalError, Node, Var
from mcp_abacus.expr.value import EvalContext, InexactHandling, Mode, Value

# Adaptive-Simpson recursion-depth backstop (40.18). With ``eps`` floored to the mode's
# own representable precision (one ULP in fixed-point, a small constant in float/rational),
# every smooth integrand converges in a handful of levels; this cap only guards a
# pathological integrand from unbounded subdivision (it bounds any one branch's depth).
_MAX_DEPTH = 20


def dispatch_special_form(name: str, ctx: EvalContext, args: tuple[Node, ...]) -> Value:
    """Evaluate the special form ``name`` against the unevaluated ``args`` (40.18).

    The entry FuncCall._evaluate defers to. One form today (``integral``); the table grows
    as the solver-adjacent family lands (diff/40.17, sum-product/40.19).
    """
    if name == "integral":
        return _integral(ctx, args)
    raise EvalError(f"unknown special form: {name!r}", line=0)


def _integral(ctx: EvalContext, args: tuple[Node, ...]) -> Value:
    """Definite integral ``integral(expr, var, a, b)`` over ``[a, b]`` (40.18).

    ``expr`` is the unevaluated integrand, ``var`` a bare NAME (the integration variable),
    ``a``/``b`` the bounds (ordinary expressions, evaluated up front). Adaptive Simpson in
    the active mode; the result is always inexact. A transcendental integrand in rational
    mode refuses at the sample (sin/log/... refuse on non-zero rationals everywhere) — the
    same consistent stance, surfaced as the integrand's own line-tagged EvalError.
    """
    expr, var_node, a_node, b_node = args

    # The variable must be a bare NAME: a Var (a constant like pi/e parses to a nullary
    # FuncCall, a literal to a Number — neither is a name to bind a sample to).
    if not isinstance(var_node, Var):
        raise EvalError("integral's variable (2nd argument) must be a name", line=var_node.line)
    name = var_node.name
    # A dummy variable the form rebinds cannot also be a name the integrand assigns (the
    # solver's validate_unknown stance). Unreachable through the parser today — an argument
    # is an expression, never an assignment — but guards a directly built tree.
    if name in expr.assigned_names():
        raise EvalError(
            f"integration variable {name!r} is assigned by the integrand", line=var_node.line
        )

    a = a_node._walk(ctx)
    b = b_node._walk(ctx)

    def sample(x: Value) -> Value:
        # Re-evaluate the integrand with ``name`` bound to ``x`` (40.18). The child store is
        # seeded from the run's so the integrand still reads outer bindings, while ``name``
        # shadows any outer binding of the same name. CONTINUE_AND_REPORT for the inner walk
        # so a sample's inexactness never aborts here — the abort, if requested, fires once
        # on integral's own inexact result at the outer FuncCall node.
        store = ctx.variables.copy()
        store.set(name, x)
        return expr.evaluate(
            ctx.mode,
            ctx.min_fixed_point_precision,
            now_ns=ctx.now_ns,
            variables=store,
            inexact_handling=InexactHandling.CONTINUE_AND_REPORT,
        )

    # Small mode-faithful constants for the rule, built once.
    two = Value.from_real(2, ctx.mode, 0)
    four = Value.from_real(4, ctx.mode, 0)
    six = Value.from_real(6, ctx.mode, 0)
    fifteen = Value.from_real(15, ctx.mode, 0)
    eps = _tolerance(ctx, a, b)
    limit = fifteen.mul(eps)  # the |S2 - S1| <= 15*eps convergence threshold (Lyness)

    def midpoint(lo: Value, hi: Value) -> Value:
        return lo.add(hi).div(two)

    def simpson(flo: Value, fmid: Value, fhi: Value, lo: Value, hi: Value) -> Value:
        # (hi - lo)/6 * (flo + 4*fmid + fhi).
        weighted = flo.add(four.mul(fmid)).add(fhi)
        return hi.sub(lo).mul(weighted).div(six)

    def recurse(
        lo: Value, hi: Value, flo: Value, fhi: Value, fmid: Value, whole: Value, depth: int
    ) -> Value:
        mid = midpoint(lo, hi)
        lmid, rmid = midpoint(lo, mid), midpoint(mid, hi)
        flmid, frmid = sample(lmid), sample(rmid)
        left = simpson(flo, flmid, fmid, lo, mid)
        right = simpson(fmid, frmid, fhi, mid, hi)
        delta = left.add(right).sub(whole)  # S2 - S1
        if depth <= 0 or delta.abs_()._compare(limit, "integral") <= 0:
            # Richardson extrapolation: the converged estimate plus delta/15.
            return left.add(right).add(delta.div(fifteen))
        return recurse(lo, mid, flo, fmid, flmid, left, depth - 1).add(
            recurse(mid, hi, fmid, fhi, frmid, right, depth - 1)
        )

    if a._compare(b, "integral") == 0:
        # Zero-width interval: the integral is 0 (still flagged inexact, per 40.18).
        return _as_inexact(a.sub(b))

    a_mid = midpoint(a, b)
    fa, fb, fmid = sample(a), sample(b), sample(a_mid)
    whole = simpson(fa, fmid, fb, a, b)  # the one-panel estimate the recursion refines
    return _as_inexact(recurse(a, b, fa, fb, fmid, whole, _MAX_DEPTH))


def _tolerance(ctx: EvalContext, a: Value, b: Value) -> Value:
    """The per-interval convergence tolerance ``eps`` for adaptive Simpson (40.18).

    Floored to the active mode's own representable precision so the recursion always
    converges (a zero tolerance would force every branch to the depth cap): one ULP at the
    bounds' working scale in fixed-point, a small absolute constant in float and rational.
    """
    match ctx.mode:
        case Mode.FLOATING_POINT:
            return Value.from_real(1e-8, Mode.FLOATING_POINT, 0)
        case Mode.RATIONAL:
            return Value.from_real(Fraction(1, 10**12), Mode.RATIONAL, 0)
        case Mode.FIXED_POINT:
            scale = max(a.payload.decimals, b.payload.decimals)  # type: ignore[union-attr]
            return Value.from_real(Fraction(1, 10**scale), Mode.FIXED_POINT, scale)
        case _:  # pragma: no cover - mirrors Value's mode exhaustiveness
            raise ValueError(f"unsupported mode: {ctx.mode!r}")


def _as_inexact(value: Value) -> Value:
    """Re-stamp ``value`` as inexact (40.18): a quadrature only approximates the integral.

    Even when the accumulating arithmetic stayed exact (e.g. Simpson is exact for a cubic
    in rational mode), the result stands for the true integral, which it merely
    approximates — so the honest flag is inexact, and the clean-residual ``error`` (a
    rounding diagnostic) does not apply.
    """
    return Value(value.mode, value.payload, exact=False)
