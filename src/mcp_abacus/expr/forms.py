# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Special forms — call functions whose arguments are NOT all evaluated operands (TODO 40).

The third callable kind beside the operand-methods (_FUNCS) and the nullaries
(_NULLARY_FUNCS) in expr.nodes. A special form receives the run EvalContext and its
arguments' UNEVALUATED subtrees, because some arguments are an unevaluated sub-program
and a free variable NAME rather than values — the solver's shape, brought inside the
expression language. ``integral(expr, var, a, b)`` (40.18) is the first: it re-evaluates
``expr`` at many sample points with ``var`` rebound to each, the way solver.py drives an
expression while searching. ``diff(expr, var, at)`` (40.17) is its sibling — a numerical
derivative, the same rebind-and-sample shape over a tight finite-difference stencil.
``sum(i, lo, hi, expr)`` / ``product(i, lo, hi, expr)`` (40.19) are the Σ/Π folds — the same
rebind-the-body shape, but over INTEGER steps from ``lo`` to ``hi`` inclusive rather than a
continuous interval. ``map(vector, name, body)`` (40.24) is the same rebind-the-body shape once
more, but over the ELEMENTS of an already-built vector, and it RETURNS a vector rather than a
scalar — the first form that both consumes and produces one (reusing factor's producer, 40.23).
``residual_sum_squares(expr, var, xs, ys)`` (40.25) folds the same rebind-the-model shape over
two PAIRED data vectors walked in lockstep, accumulating the squared gap
``Σ(ys_i - model(xs_i))**2`` — a least-squares cost, the caller-supplied-model analogue of
curve_fit's residual error.

This is a SIBLING engine to solver.py: it imports expr.nodes (Node.evaluate, the AST) to
re-run the integrand, an edge value.py cannot carry — which is why these live here and not
as Value methods. expr.nodes imports THIS module only at call time (FuncCall._evaluate),
so the back-edge never exists at module load.

Everything stays in the ACTIVE mode: a definite integral is just +, *, /, and one
comparison, all of which Value already does mode-faithfully, so the quadrature accumulates
with Value arithmetic — no float shadow (unlike the solver, whose golden-section / parabola
math is intrinsically floating-point). The integral result is flagged inexact in EVERY mode
(40.18), because Simpson's rule only APPROXIMATES the true integral — not because of any
conversion. The same holds for ``diff`` (40.17): a finite-difference quotient only
APPROXIMATES the true derivative, so it too is inexact in every mode.

``sum``/``product`` (40.19) and ``residual_sum_squares`` (40.25) are the EXCEPTION: a finite
fold is EXACT — it adds and multiplies the body's own Values with no truncation of its own — so
it keeps the per-mode exactness of the body. ``sum`` is repeated ``+`` (exact in every mode,
like ``sum_``); ``product`` is repeated ``*`` (may round to the covering fixed-point scale,
exact in rational, rounds in float); ``residual_sum_squares`` is subtract + square + add, so it
follows the same stance as ``product`` (the square may round in fixed-point/float, stays exact
in rational). Either way the inexactness is the body's or the arithmetic's, never the form's. A
finite range can still be huge, so ``sum``/``product``'s term count is capped (a DoS guard,
mirroring factorial's cap); ``residual_sum_squares`` folds over the caller's data, already
bounded by the vector length.
"""

from collections.abc import Callable
from fractions import Fraction

from mcp_abacus.expr.nodes import EvalError, Node, Var
from mcp_abacus.expr.value import EvalContext, InexactHandling, Mode, Value

# Adaptive-Simpson recursion-depth backstop (40.18). With ``eps`` floored to the mode's
# own representable precision (one ULP in fixed-point, a small constant in float/rational),
# every smooth integrand converges in a handful of levels; this cap only guards a
# pathological integrand from unbounded subdivision (it bounds any one branch's depth).
_MAX_DEPTH = 20

# Term-count ceiling for the sum/product folds (40.19). A finite range is finite by
# construction, but ``sum(i, 1, 1000000000, i)`` would still grind the server to a halt, so
# a range wider than this REFUSES rather than loops — the same DoS guard factorial's 1000
# cap is, sized far higher because a series legitimately wants many terms.
_MAX_RANGE = 100000


def dispatch_special_form(name: str, ctx: EvalContext, args: tuple[Node, ...]) -> Value:
    """Evaluate the special form ``name`` against the unevaluated ``args`` (40.18/40.19).

    The entry FuncCall._evaluate defers to. The solver-adjacent family: ``integral`` (40.18)
    and ``diff`` (40.17) sample over a continuous interval; ``sum``/``product`` (40.19) fold
    over integer steps.
    """
    if name == "integral":
        return _integral(ctx, args)
    if name == "diff":
        return _diff(ctx, args)
    if name == "sum":
        return _range_fold(ctx, args, "sum", Value.add, 0)
    if name == "product":
        return _range_fold(ctx, args, "product", Value.mul, 1)
    if name == "map":
        return _map(ctx, args)
    if name == "residual_sum_squares":
        return _residual_sum_squares(ctx, args)
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


def _diff(ctx: EvalContext, args: tuple[Node, ...]) -> Value:
    """Numerical derivative ``diff(expr, var, at)`` w.r.t. ``var`` at the point ``at`` (40.17).

    ``expr`` is the unevaluated expression, ``var`` a bare NAME (the variable to differentiate
    against), ``at`` the point (an ordinary expression, evaluated up front). A 4th-order
    five-point central-difference stencil in the active mode; the result is ALWAYS inexact (a
    finite-difference quotient only approximates the derivative). A transcendental ``expr`` in
    rational mode refuses at the sample, the same consistent stance integral takes — surfaced
    as the expression's own line-tagged EvalError.
    """
    expr, var_node, at_node = args

    # The variable must be a bare NAME — same stance as integral (a constant like pi/e parses
    # to a nullary FuncCall, a literal to a Number; neither is a name to bind a sample to).
    if not isinstance(var_node, Var):
        raise EvalError("diff's variable (2nd argument) must be a name", line=var_node.line)
    name = var_node.name
    # A dummy variable the form rebinds cannot also be a name the expression assigns (the
    # solver's validate_unknown stance; mirrors integral). Unreachable through the parser today.
    if name in expr.assigned_names():
        raise EvalError(
            f"differentiation variable {name!r} is assigned by the expression", line=var_node.line
        )

    at = at_node._walk(ctx)

    def sample(x: Value) -> Value:
        # Re-evaluate ``expr`` with ``name`` bound to ``x`` (40.17) — same child-store /
        # CONTINUE_AND_REPORT handling as integral: the expression still reads outer bindings,
        # ``name`` shadows any outer binding of the same name, and a sample's inexactness never
        # aborts here (the abort, if any, fires once on diff's own inexact result outside).
        store = ctx.variables.copy()
        store.set(name, x)
        return expr.evaluate(
            ctx.mode,
            ctx.min_fixed_point_precision,
            now_ns=ctx.now_ns,
            variables=store,
            inexact_handling=InexactHandling.CONTINUE_AND_REPORT,
        )

    eight = Value.from_real(8, ctx.mode, 0)
    twelve = Value.from_real(12, ctx.mode, 0)
    two = Value.from_real(2, ctx.mode, 0)
    h = _step(ctx, at)
    two_h = two.mul(h)

    # Five-point central difference: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h).
    # Higher order than the 3-point quotient (truncation O(h^4), exact for polynomials up to
    # degree 4), so it holds up where low-precision fixed-point arithmetic would otherwise lose
    # the derivative to cancellation. Written as f(x-2h) - f(x+2h) + 8*(f(x+h) - f(x-h)).
    inner = eight.mul(sample(at.add(h)).sub(sample(at.sub(h))))
    numerator = sample(at.sub(two_h)).sub(sample(at.add(two_h))).add(inner)
    return _as_inexact(numerator.div(twelve.mul(h)))


def _range_fold(
    ctx: EvalContext,
    args: tuple[Node, ...],
    what: str,
    op: Callable[[Value, Value], Value],
    identity: int,
) -> Value:
    """The shared Σ/Π range fold behind ``sum``/``product`` (40.19).

    ``sum(i, lo, hi, expr)`` / ``product(i, lo, hi, expr)``: ``i`` is a bare NAME (the index),
    ``lo``/``hi`` the inclusive integer bounds (ordinary expressions, evaluated and read as
    integers up front), ``expr`` the unevaluated body re-evaluated once per integer ``i`` in
    ``[lo, hi]``. The body's term Values are combined with ``op`` (``Value.add`` for Σ,
    ``Value.mul`` for Π) starting from the first term — so unlike integral/diff the fold is
    EXACT (no truncation of its own) and inherits the body's per-mode story: Σ stays exact
    like repeated ``+``, Π may round to the covering fixed-point scale like repeated ``*``.
    An EMPTY range (``lo > hi``) yields the fold's ``identity`` (0 for Σ, 1 for Π) at scale 0.
    A transcendental body in rational mode refuses at the term, the same stance integral takes.
    """
    index_node, lo_node, hi_node, body = args

    # The index must be a bare NAME — same stance as integral/diff's variable (a constant
    # like pi/e parses to a nullary FuncCall, a literal to a Number; neither names a term).
    if not isinstance(index_node, Var):
        raise EvalError(f"{what}'s index (1st argument) must be a name", line=index_node.line)
    name = index_node.name
    # A dummy the fold rebinds cannot also be a name the body assigns (mirrors integral/diff).
    # Unreachable through the parser today — an argument is an expression, never an assignment.
    if name in body.assigned_names():
        raise EvalError(f"{what} index {name!r} is assigned by the body", line=index_node.line)

    # The bounds are integers — the fold steps by one integer at a time (``_as_integer``
    # REFUSES a fractional fixed-point value, a rational with denominator != 1, or a non-whole
    # float, the same exact-or-refuse gate gcd/lcm/factorial use; its ArithmeticError surfaces
    # line-tagged at the call node).
    lo = lo_node._walk(ctx)._as_integer(f"{what} bounds")
    hi = hi_node._walk(ctx)._as_integer(f"{what} bounds")

    terms = hi - lo + 1
    if terms <= 0:
        # Empty range: the additive/multiplicative identity, a whole number at scale 0.
        return Value.from_real(identity, ctx.mode, 0)
    if terms > _MAX_RANGE:
        raise EvalError(
            f"{what} range exceeds {_MAX_RANGE} terms ({terms} requested)", line=index_node.line
        )

    def term(k: int) -> Value:
        # Re-evaluate the body with ``name`` bound to the integer ``k`` (40.19) — same
        # child-store / CONTINUE_AND_REPORT handling as integral/diff: the body still reads
        # outer bindings, ``name`` shadows any outer binding of the same name, and a term's
        # inexactness never aborts mid-fold (the abort, if requested, fires once on the fold's
        # own result at the outer FuncCall node). The index is materialised at the precision
        # FLOOR so a raised floor carries through, exactly as integral's bounds do.
        store = ctx.variables.copy()
        store.set(name, Value.from_real(k, ctx.mode, ctx.min_fixed_point_precision))
        return body.evaluate(
            ctx.mode,
            ctx.min_fixed_point_precision,
            now_ns=ctx.now_ns,
            variables=store,
            inexact_handling=InexactHandling.CONTINUE_AND_REPORT,
        )

    acc = term(lo)
    for k in range(lo + 1, hi + 1):
        acc = op(acc, term(k))
    return acc


def _map(ctx: EvalContext, args: tuple[Node, ...]) -> Value:
    """Element-wise transform ``map(vector, name, body)`` — a new VECTOR (40.24).

    ``vector`` is an ordinary expression evaluated up front (it MUST be a vector — a
    scalar is refused, map is vector-in/vector-out), ``name`` a bare NAME (the element
    dummy), ``body`` the unevaluated expression re-evaluated once per element with
    ``name`` rebound to it. The results collect into a same-length VECTOR in the run's
    element mode — the FIRST function that both CONSUMES and PRODUCES a vector, reusing
    factor's ``Value.vector`` producer plumbing (40.23).

    Higher-order like sum/product (40.19): map itself only DISPATCHES — it adds no
    arithmetic and stamps no verdict of its own. Exactness is the BODY's, per element:
    ``Value.vector`` folds the elements' flags, so the result is exact iff every mapped
    element is (an EMPTY source maps to an empty vector, vacuously exact). A transcendental
    body in rational mode refuses at the element, the same stance the other forms take.
    """
    vector_node, name_node, body = args

    # The element variable must be a bare NAME — same stance as integral/diff/sum (a
    # constant like pi/e parses to a nullary FuncCall, a literal to a Number; neither is a
    # name to bind an element to).
    if not isinstance(name_node, Var):
        raise EvalError("map's variable (2nd argument) must be a name", line=name_node.line)
    name = name_node.name
    # A dummy the map rebinds cannot also be a name the body assigns (mirrors the other
    # forms). Unreachable through the parser today — an argument is an expression, never an
    # assignment — but guards a directly built tree.
    if name in body.assigned_names():
        raise EvalError(f"map variable {name!r} is assigned by the body", line=name_node.line)

    source = vector_node._walk(ctx)
    if source.mode is not Mode.VECTOR:
        # map transforms a SERIES; a scalar has no elements to walk. Refuse rather than
        # silently wrap it in a 1-element vector (the vector-in/vector-out contract, 40.24).
        raise EvalError("map's first argument must be a vector", line=vector_node.line)

    results = []
    for element in source.payload.elements:  # type: ignore[union-attr]
        # Re-evaluate the body with ``name`` bound to the element (40.24) — same child-store /
        # CONTINUE_AND_REPORT handling as sum/product: the body still reads outer bindings,
        # ``name`` shadows any outer binding of the same name, and an element's inexactness
        # never aborts mid-map (the abort, if requested, fires once on map's own vector result
        # at the outer FuncCall node).
        store = ctx.variables.copy()
        store.set(name, element)
        results.append(
            body.evaluate(
                ctx.mode,
                ctx.min_fixed_point_precision,
                now_ns=ctx.now_ns,
                variables=store,
                inexact_handling=InexactHandling.CONTINUE_AND_REPORT,
            )
        )
    # Pack straight into a VECTOR in the run's element mode, exactly as factor does (40.23);
    # a body that produced a vector makes this refuse (no nesting, per Value.vector).
    return Value.vector(results, ctx.mode)


def _residual_sum_squares(ctx: EvalContext, args: tuple[Node, ...]) -> Value:
    """Least-squares cost ``residual_sum_squares(expr, var, xs, ys)`` of a model (40.25).

    ``Σ_i (ys_i - expr[var := xs_i])**2`` — the sum of squared residuals (RSS/SSE) of a
    caller-supplied model against PAIRED observed data. ``expr`` is the unevaluated model,
    ``var`` a bare NAME (its free variable), ``xs``/``ys`` two EQUAL-LENGTH, NON-EMPTY data
    vectors (ordinary expressions, evaluated up front). The two series walk in LOCKSTEP: for
    each pair the model is re-evaluated with ``var`` bound to ``xs_i``, subtracted from
    ``ys_i``, squared, and folded into the running sum. This is the caller-supplied-model
    analogue of curve_fit's residual error (which only ranks its built-in forms).

    RETURNS a SCALAR. Like ``sum`` (40.19) the fold adds no truncation of its own and stamps
    no verdict — the per-mode story is the underlying arithmetic's (subtract + square + add):
    EXACT in rational, MAY ROUND in fixed-point/float where the square leaves the grid (the
    avg/sum stance, NOT sqrt's — there is no root). A transcendental model in rational mode
    refuses at the point, the same stance the other forms take. DOMAIN: ``xs``/``ys`` must be
    two equal-length, non-empty vectors — unequal lengths (unpaired data) and empty data both
    REFUSE, mirroring covariance (40.13).
    """
    expr, var_node, xs_node, ys_node = args

    # The model variable must be a bare NAME — same stance as integral/diff/map (a constant
    # like pi/e parses to a nullary FuncCall, a literal to a Number; neither names a point).
    if not isinstance(var_node, Var):
        raise EvalError(
            "residual_sum_squares's variable (2nd argument) must be a name", line=var_node.line
        )
    name = var_node.name
    # A dummy the fold rebinds cannot also be a name the model assigns (mirrors the other
    # forms). Unreachable through the parser today — an argument is an expression, never an
    # assignment — but guards a directly built tree.
    if name in expr.assigned_names():
        raise EvalError(
            f"model variable {name!r} is assigned by the model expression", line=var_node.line
        )

    xs_value = xs_node._walk(ctx)
    ys_value = ys_node._walk(ctx)
    if xs_value.mode is not Mode.VECTOR or ys_value.mode is not Mode.VECTOR:
        raise EvalError(
            "residual_sum_squares's data (3rd and 4th arguments) must be two vectors",
            line=xs_node.line,
        )
    xs = xs_value.payload.elements  # type: ignore[union-attr]
    ys = ys_value.payload.elements  # type: ignore[union-attr]
    if len(xs) != len(ys):
        raise EvalError(
            "residual_sum_squares requires two equal-length data vectors", line=xs_node.line
        )
    if not xs:
        raise EvalError("residual_sum_squares of empty data is undefined", line=xs_node.line)

    def squared_residual(x: Value, y: Value) -> Value:
        # Re-evaluate the model with ``name`` bound to the point ``x`` (40.25) — same child-store
        # / CONTINUE_AND_REPORT handling as sum/product/map: the model still reads outer bindings,
        # ``name`` shadows any outer binding of the same name, and a point's inexactness never
        # aborts mid-fold (the abort, if requested, fires once on the scalar result at the outer
        # FuncCall node). The observed ``y`` and the model share the run mode, so ``sub`` is
        # same-mode; the square is a plain ``mul`` (may round in fixed-point/float, exact rational).
        store = ctx.variables.copy()
        store.set(name, x)
        predicted = expr.evaluate(
            ctx.mode,
            ctx.min_fixed_point_precision,
            now_ns=ctx.now_ns,
            variables=store,
            inexact_handling=InexactHandling.CONTINUE_AND_REPORT,
        )
        gap = y.sub(predicted)
        return gap.mul(gap)

    total = squared_residual(xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:], strict=True):
        total = total.add(squared_residual(x, y))
    return total


def _step(ctx: EvalContext, x: Value) -> Value:
    """The finite-difference step ``h`` for ``diff`` at the point ``x`` (40.17).

    Balances the stencil's two competing errors — truncation (shrinks with ``h``) against
    round-off from cancellating nearby samples (grows as ``h`` shrinks), so ``h`` tracks each
    mode's working precision. Float and rational take a small fixed step (their grids are fine
    enough that the 4th-order truncation dominates and stays negligible); fixed-point scales
    the step to the point's own decimal scale, roughly the 5th-root-of-ULP optimum for the
    five-point rule, never finer than 0.1 of a unit so the differenced samples keep enough
    significant digits — and exactly one unit at scale 0 (integers), the only step on that grid.
    """
    match ctx.mode:
        case Mode.FLOATING_POINT:
            return Value.from_real(1e-5, Mode.FLOATING_POINT, 0)
        case Mode.RATIONAL:
            # Scale 6, NOT 0: a rational stores at the given scale, so scale 0 would round
            # the step to 0 (and divide by zero). 10**-6 is exact at 6 decimals — the step
            # then stays an exact fraction the rational arithmetic carries losslessly.
            return Value.from_real(Fraction(1, 10**6), Mode.RATIONAL, 6)
        case Mode.FIXED_POINT:
            scale = x.payload.decimals  # type: ignore[union-attr]
            k = max(1, round(scale / 5)) if scale >= 1 else 0
            return Value.from_real(Fraction(1, 10**k), Mode.FIXED_POINT, scale)
        case _:  # pragma: no cover - mirrors Value's mode exhaustiveness
            raise ValueError(f"unsupported mode: {ctx.mode!r}")


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
