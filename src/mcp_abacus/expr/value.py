# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""The universal Value class — item 19's real design (empty shell).

DESIGN GOAL: Value is the SINGLE chokepoint for type behaviour. Adding a new
numeric type must be a fully local, two-step change — (1) add its member to the
type enum (tracked in TODO 19), (2) implement that member's branch in each
operation here — after which the type is supported everywhere, because every
other part of the engine is generic over the enum and NEVER branches on type
itself. ``mode`` is therefore a plain enum: one member per fully-specified
concrete type (e.g. SIGNED_INTEGER_64, not INTEGER + signed + width). No
parametric types, no ModeSpec. Any quantity that varies per value (a
fixed-point scale, a decimal's digits) rides in the payload; any policy
(rounding, division rule) is fixed code in that type's branch. Each operation
dispatches with a per-operation match over the enum, one case per type.

This is the home of ALL arithmetic for the expression engine. It grows
alongside the minimal slice in ``values.py`` (the two parameter-free modes
"rational" + "floating-point" shipped under TODO 18); that slice keeps the engine
working today, and this class supersedes it once the operators below are
filled in (19.3.1-19.3.9) and the callers are rewired.

The operator methods are intentionally STUBS for now. This docstring records
the standing design DECISIONS — these are not tasks that ever complete, they
are the contract every future change must honor:

  from_lexeme contract.  from_lexeme(lexeme, mode) interprets the SOURCE STRING
        in the mode's own representation (this is where 17.2.2's keep-the-lexeme
        rationale lands). It must handle every 20.1.3 literal form, including
        the base-prefixed integers 0xFF / 0b1010 / 0o17 (20.4) — a base-prefixed
        lexeme is an INTEGER literal, valid in EVERY mode as that integer.

  No mode mixing.  A binary op requires BOTH operands in the SAME mode,
        otherwise it raises. There is no silent promotion between modes —
        faithfulness over convenience. (And no operand promotion from plain
        Python numbers either: operators take Values, nothing else.)

  ``**`` is POWER.  There is no __pow__ to mislead. The ** -> power
        translation happens in BinOp.evaluate (18.2) and is routed through the
        named pow() method below — never through a Python dunder.

  Immutable (frozen).  Every operation returns a NEW Value; nothing mutates in
        place. Each Value carries its own ``exact: bool``, and ops propagate it
        (result exact only if every operand was) — the value's own exactness,
        tracked almost for free.

  Per-operation dispatch.  Each operation runs its OWN match over the type enum,
        one case per type — NOT one giant switch handling every op x type combo.
        Adding a type means adding a case to each operation's match.

  Mode-agnostic domain errors.  Domain errors (division by zero,
        overflow-as-error, the f16 struct OverflowError caveat 10.L.2) raise
        exceptions carrying NO line information; BinOp.evaluate (18.4) is what
        attaches node.line.

NO operator overloading (decision 2026-06-12): the ops we implement may
grow complicated and need MORE arguments than a Python dunder's fixed signature
allows (per-op rounding mode, precision context, overflow policy, returning a
carry/flags alongside the result, ...). Hence the explicit named methods below.

OPEN faithfulness questions (provisional in the values.py slice, to settle as
the real ops land here):
  - floating-point division by zero RAISES (follows Python); IEEE-754 would give
    ±inf. Pick the faithful rule (relates to 10.2.3).
  - floating-point exactness is unconditionally False — conservative, since e.g.
    1.0 + 2.0 IS exact; refine via the ``exact`` flag.
  - over-range int -> float is inconsistent: a huge 0x literal raises
    OverflowError, but float("1e999") yields inf. Choose ONE faithful rule.

STILL TO PLAN, deliberately absent from this shell: which concrete modes to
build after floating-point. The mode descriptor question is SETTLED (2026-06-12):
``mode`` is the plain ``Mode`` enum below — one member per concrete type, no
ModeSpec dataclass — and floating-point is the first member built (19.1.1).
"""

import functools
import math
import operator
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction


class NotRepresentableError(ArithmeticError):
    """The exact result cannot be represented in the current mode.

    An ArithmeticError subclass so BinOp.evaluate catches it and attaches the
    failing node's line (the "mode-agnostic domain errors" contract above);
    raised here carrying NO position info. Python's own ZeroDivisionError is
    likewise an ArithmeticError, so division by zero propagates the same way.
    """


class Mode(Enum):
    """The concrete numeric type a Value holds — one member per fully-specified
    type (19.1), e.g. FLOATING_POINT, not INTEGER + signed + width.

    Adding a type is step (1) of the two-step change: add its member here, then
    implement its branch in each operation. The engine is generic over this enum
    and never branches on type outside Value's per-operation dispatch.

    FLOATING_POINT, FIXED_POINT and RATIONAL exist today; 19.1.3-19.1.6/19.1.8 each
    add one more.
    """

    FLOATING_POINT = "floating-point"  # IEEE-754 double — C's "double" (19.1.1)
    FIXED_POINT = "fixed-point"  # scaled integer; exact, money/ERC-20-safe (19.1.2)
    RATIONAL = "rational"  # exact numerator/denominator; no irrationals (19.1.7)
    COMPLEX = "complex"  # a + b*i, each part a fixed-point scaled integer (19.1.8)
    VECTOR = "vector"  # internal 1-D container; elements carry the real (scalar) mode (19.1.10)


# One terse line per Mode, for the language-help `types` section. Co-located with
# the enum on purpose: it is the SINGLE source for a type's human description, so
# adding a Mode member without a line here is a hard KeyError when help renders —
# the text can never silently drift from the supported set.
MODE_HELP: dict[Mode, str] = {
    Mode.FLOATING_POINT: "IEEE-754 double; ~15-17 significant digits; most decimals inexact",
    Mode.FIXED_POINT: "scaled integer (mantissa x 10^-decimals); exact; money / ERC-20 safe",
    Mode.RATIONAL: "exact numerator/denominator; no irrationals",
    Mode.COMPLEX: "a + b*i over two fixed-point parts; exact + - *, rounds / and transcendentals",
    # VECTOR is an internal container, NOT a selectable mode — kept here only to honour the
    # one-line-per-Mode contract; `selectable_modes()` filters it out of the `types` help.
    Mode.VECTOR: "internal one-dimensional vector of values (built with [a, b, …]); not selectable",
}


def selectable_modes() -> list[Mode]:
    """The modes a caller may pick for a calculation — every Mode except VECTOR.

    VECTOR is an INTERNAL container type (19.1.10): a vector only ever arises from a
    ``[a, b, …]`` literal inside some chosen SCALAR mode, never as the mode of a whole
    calculation (``2 + 2`` has no meaning "in vector mode"). So it is excluded
    everywhere a mode is offered to the caller — `resolve_mode` refuses it, the help
    `types` section omits it, and a tool's "valid modes" error never suggests it.
    """
    return [m for m in Mode if m is not Mode.VECTOR]


# Input-only spellings the caller may use for a mode, each resolved to its
# canonical Mode before dispatch (23.6/24.2). The canonical Mode.value is the
# single name help and output report; these are forgiving aliases for what an AI
# is likely to type as its first guess, so a plausible name resolves in turn one
# instead of erroring into a retry. Never surfaced as separate types.
#
# NOTE: "decimal" maps to fixed-point because that is today's exact-decimal type.
# If a true decimal Mode (SA.2.1) ever lands, this alias must move to it.
MODE_ALIASES: dict[str, Mode] = {
    "float64": Mode.FLOATING_POINT,
    "double": Mode.FLOATING_POINT,
    "float": Mode.FLOATING_POINT,
    "ieee754": Mode.FLOATING_POINT,
    "fraction": Mode.RATIONAL,
    "frac": Mode.RATIONAL,
    "decimal": Mode.FIXED_POINT,
}


def resolve_mode(name: str) -> Mode:
    """Resolve a mode name or alias to its Mode; raise ValueError if neither.

    Tries the canonical names first (Mode's own values), then the aliases; the
    ValueError from an unknown name carries through unchanged so callers can list
    the valid modes (23.5/23.6).
    """
    try:
        mode = Mode(name)
    except ValueError:
        if name in MODE_ALIASES:
            return MODE_ALIASES[name]
        raise
    if mode is Mode.VECTOR:
        # VECTOR is internal-only (see selectable_modes): treat "vector" exactly like an
        # unknown name so a caller cannot select it, and no alias ever resolves to it.
        raise ValueError(f"{name!r} is not a valid Mode")
    return mode


class InexactHandling(Enum):
    """What a calculation does the moment an operation's result is inexact (35.2).

    Supplied by the CALLER, carried on the EvalContext and threaded down the whole
    evaluate walk, so it selects a run-wide policy rather than a per-operation one.
    Two members today:

    CONTINUE_AND_REPORT (default, 35.2.1) is the historical behaviour — compute,
    never reject, and let the exact/inexact verdict surface in the reply. ABORT_ON_
    INEXACT (35.2.2) instead unwinds the calculation as soon as any value is
    inexact, raising a diagnostic that names WHERE (the source line + the offending
    sub-expression) and WHY (the magnitude and class of the inexactness) so the
    caller who asked for it learns precisely what went inexact.
    """

    CONTINUE_AND_REPORT = "continue-and-report"  # 35.2.1 — compute and report (default)
    ABORT_ON_INEXACT = "abort-on-inexact"  # 35.2.2 — raise the moment a value is inexact


# One terse line per InexactHandling member, the single source for its human
# description (mirrors MODE_HELP). A missing entry is a hard KeyError wherever the
# help renders, so the text can never drift from the supported set.
INEXACT_HANDLING_HELP: dict[InexactHandling, str] = {
    InexactHandling.CONTINUE_AND_REPORT: "compute and report the verdict; never reject (default)",
    InexactHandling.ABORT_ON_INEXACT: "abort with a diagnostic the moment any result is inexact",
}


# Forgiving input spellings the caller may use, each resolved to a canonical
# member (mirrors MODE_ALIASES): the names an AI is likely to type as a first
# guess, so a plausible value resolves in turn one instead of erroring into a retry.
INEXACT_HANDLING_ALIASES: dict[str, InexactHandling] = {
    "continue": InexactHandling.CONTINUE_AND_REPORT,
    "report": InexactHandling.CONTINUE_AND_REPORT,
    "continue-and-report": InexactHandling.CONTINUE_AND_REPORT,
    "default": InexactHandling.CONTINUE_AND_REPORT,
    "abort": InexactHandling.ABORT_ON_INEXACT,
    "abort-on-inexact": InexactHandling.ABORT_ON_INEXACT,
    "strict": InexactHandling.ABORT_ON_INEXACT,
    "exact-only": InexactHandling.ABORT_ON_INEXACT,
    "require-exact": InexactHandling.ABORT_ON_INEXACT,
}


def resolve_inexact_handling(name: str) -> InexactHandling:
    """Resolve an inexact-handling name or alias to its member; raise ValueError if neither.

    Tries the canonical names first (the enum's own values), then the aliases; the
    ValueError from an unknown name carries through unchanged so callers can list
    the valid choices — the same shape as ``resolve_mode``.
    """
    try:
        return InexactHandling(name)
    except ValueError:
        if name in INEXACT_HANDLING_ALIASES:
            return INEXACT_HANDLING_ALIASES[name]
        raise


class UndefinedVariableError(LookupError):
    """A variable was read before any assignment bound it (30.1).

    Raised by ``VariableStore.get`` on a name that was never set, carrying that
    name so the referencing Var node (30.6) can re-raise it as an EvalError with
    the source line. A ``LookupError`` — the natural base for a missing key —
    deliberately NOT an ArithmeticError, so the evaluate walk's domain-error
    wrapping leaves it alone and the Var node positions it itself.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"undefined variable: {name}")
        self.name = name


class VariableStore:
    """Named values for one evaluation run: variable name (str) -> Value (30.1).

    A thin wrapper over a dict, carried on the EvalContext (30.2) so a run's
    assignments (``x = expr``) stay visible to later references (a bare ``x``).
    ``set`` binds or overwrites; ``get`` on a name that was never assigned RAISES
    (UndefinedVariableError) rather than inventing a default — reading an unset
    variable is a user error, not a zero.

    Mutable state, so a plain slotted class rather than a frozen dataclass; the
    frozen EvalContext holds it by reference and its contents mutate as the run
    assigns.
    """

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, Value] = {}

    def set(self, name: str, value: "Value") -> None:
        """Bind ``name`` to ``value``, overwriting any previous binding."""
        self._values[name] = value

    def get(self, name: str) -> "Value":
        """Return the Value bound to ``name``; raise UndefinedVariableError if unset."""
        try:
            return self._values[name]
        except KeyError:
            raise UndefinedVariableError(name) from None

    def copy(self) -> "VariableStore":
        """A shallow clone — a fresh store with the SAME bindings (40.18).

        The solver-adjacent forms (``integral``) re-evaluate a sub-program many times
        with one name rebound to each sample point; they seed a per-sample store from
        the enclosing run's store so the integrand can still read the outer bindings,
        while the bound (dummy) variable shadows any outer binding of the same name.
        Cloning keeps those per-sample rebindings from leaking back into the run's own
        store. Values are immutable, so a shallow dict copy is enough.
        """
        clone = VariableStore()
        clone._values = dict(self._values)
        return clone


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Per-run evaluation state, threaded down the evaluate walk (29.1).

    ONE instance is built at the top of a single ``Node.evaluate`` run and passed
    to every node as it is walked, so the per-run state lives in an argument
    rather than a module global. It carries the ``mode`` the run evaluates in, the
    ``min_fixed_point_precision`` floor (25.2.1), and the ``nullary_precision``
    below.

    ``nullary_precision`` (29.3) is the fixed-point scale a nullary like ``pi()``
    produces — a constant has no operand to carry a scale, so the run hands it one:
    the floor raised to the largest decimal scale of any literal in the expression,
    derived by a single pre-walk before evaluation (FIXED_POINT only — float uses
    its native constant, rational refuses the irrational, neither has a scale). It
    is unused outside fixed-point and defaults to 0.

    ``now_ns`` (28.1.2) is the single REALTIME clock reading the run was sampled
    at — integer nanoseconds since the Unix epoch — shared by every ``time()`` in
    the expression so a run sees ONE instant (``time() - time() == 0`` exactly).
    ``evaluate`` samples it once from the real clock; tests inject a fixed epoch.
    ``None`` when the run never needs it (no ``time()`` call); the only nullary
    that reads the context for anything beyond mode/scale.

    ``variables`` (30.2) is the run's VariableStore — the named-value scope an
    assignment (``x = expr``) writes and a later reference reads. A fresh, empty
    store per run by default (assignments do not leak across ``evaluate`` calls);
    being mutable it is the one field whose contents change during the walk, the
    frozen context holding it by reference.

    ``inexact_handling`` (35.2) is the caller-supplied policy for an inexact result:
    CONTINUE_AND_REPORT (the default) computes and lets the verdict surface, while
    ABORT_ON_INEXACT makes the walk raise the moment a value is inexact. Carried
    here so the policy reaches the node walk that knows the line and sub-expression.
    """

    mode: Mode
    min_fixed_point_precision: int = 0
    nullary_precision: int = 0
    now_ns: int | None = None
    variables: VariableStore = field(default_factory=VariableStore)
    inexact_handling: InexactHandling = InexactHandling.CONTINUE_AND_REPORT


@dataclass(frozen=True, slots=True)
class FixedPoint:
    """A fixed-point number as a scaled integer: value == mantissa * 10**-decimals.

    The FIXED_POINT payload (19.1.2). ``mantissa`` is an arbitrary-precision
    Python ``int`` — bignum, so no extra dependency is needed and a full-width
    ERC-20 amount (1 ETH == 10**18 wei) fits with room to spare. ``decimals`` is
    the scale: the number of fractional digits, always >= 0.

    Trailing zeros in the scale are SIGNIFICANT: FixedPoint(150, 2) (1.50) and
    FixedPoint(15, 1) (1.5) are numerically equal but carry different declared
    precision. The operators preserve this — a binary op's result scale is the
    max() of its operands' decimals (the scale that covers BOTH exactly), and a
    result that does not fit that scale is rounded half-to-even and flagged
    inexact (10.1.1).
    """

    mantissa: int
    decimals: int

    def __post_init__(self) -> None:
        # type(...) is int rejects bool (an int subclass), mirroring FLOATING_POINT.
        if type(self.mantissa) is not int:
            raise ValueError("FixedPoint mantissa must be an int")
        if type(self.decimals) is not int or self.decimals < 0:
            raise ValueError("FixedPoint decimals must be a non-negative int")


@dataclass(frozen=True, slots=True)
class Complex:
    """A complex number as a pair of fixed-point parts: value == real + imag*i.

    The COMPLEX payload (19.1.8). Each part is an independent ``FixedPoint`` — its
    own mantissa and scale — so the whole engine of fixed-point semantics (the
    covering-scale rule, half-to-even rounding, the exact signed residual) is
    reused per part rather than reinvented: every complex operation is expressed
    as fixed-point operations on the parts (see Value._complex_* below). A complex
    value is EXACT only when BOTH parts are exact, mirroring the single ``exact``
    flag every Value carries; the parts may carry different scales (the algebra
    keeps each part at the precision its own sub-expression produced).
    """

    real: FixedPoint
    imag: FixedPoint

    def __post_init__(self) -> None:
        if type(self.real) is not FixedPoint or type(self.imag) is not FixedPoint:
            raise ValueError("Complex parts must both be FixedPoint")


@dataclass(frozen=True, slots=True)
class Vector:
    """A one-dimensional vector: an ordered run of Values, all in one element mode.

    The VECTOR payload (19.1.10). VECTOR is an INTERNAL container, not a selectable
    calculation mode: a vector arises only from a ``[a, b, …]`` literal evaluated in
    some chosen SCALAR mode, and every element is a Value in THAT mode — the
    container is generic over the element type, exactly as the engine is generic
    over ``Mode``. ``element_mode`` records that shared scalar mode explicitly so an
    EMPTY vector still knows what it holds (its elements give no other clue).

    Strictly one-dimensional: an element is never itself a vector (``element_mode``
    is never VECTOR), so there is no nesting to recurse through. A vector is EXACT
    only when EVERY element is exact (an empty vector vacuously so), mirroring the
    single ``exact`` flag every Value carries.
    """

    element_mode: "Mode"
    elements: tuple["Value", ...]

    def __post_init__(self) -> None:
        if self.element_mode is Mode.VECTOR:
            raise ValueError("Vector element_mode must be a scalar mode, not VECTOR")
        for element in self.elements:
            if type(element) is not Value:
                raise ValueError("Vector elements must all be Value")
            if element.mode is not self.element_mode:
                raise ValueError(
                    f"Vector element mode {element.mode.value} != {self.element_mode.value}"
                )


def _hex_bytes(magnitude: int) -> str:
    """``magnitude`` (>= 0) as hex digits padded to whole bytes (26.8).

    Hex is always shown a whole byte at a time, so an odd digit count gets a
    leading zero: 1 -> "01", 0x1af -> "01af", 0 -> "00". The caller prepends the
    "0x" and any sign.
    """
    digits = f"{magnitude:x}"
    return digits.zfill(len(digits) + len(digits) % 2)


def _approx_decimal(frac: Fraction) -> str:
    """A bounded-precision decimal approximation of an exact fraction (26.7).

    ~24 significant digits; a trailing '…' marks a decimal the precision could not
    recover exactly (1/3 → 0.333...3…), so the approximation never poses as the
    exact value the fraction itself already shows. A terminating fraction (1/2 →
    0.5) gets no ellipsis. Shared by a rational's own approximation and the
    fixed-point error fragment (34.5.2).
    """
    with localcontext() as ctx:
        ctx.prec = 24
        approx = Decimal(frac.numerator) / Decimal(frac.denominator)
    text = str(approx)
    if Fraction(approx) != frac:
        text += "…"
    return text


def _fp_quantize(num: int, den: int, scale: int) -> tuple[int, bool]:
    """Round the exact rational ``num/den`` to a mantissa at ``scale`` decimals.

    Returns ``(mantissa, lossless)`` where ``mantissa * 10**-scale`` is the
    nearest representable value (ties to even — the IEEE/Decimal default) and
    ``lossless`` is True iff the rounding discarded nothing. ``den`` must be
    non-zero; its sign is normalised here so negative results round correctly.
    """
    if den < 0:
        num, den = -num, -den
    q, r = divmod(num * 10**scale, den)  # q floored, 0 <= r < den
    if r == 0:
        return q, True
    if 2 * r > den or (2 * r == den and q % 2 == 1):
        q += 1  # round half to even
    return q, False


def _fp_floor(a: FixedPoint, b: FixedPoint) -> int:
    """``floor(a / b)`` as a whole number; the caller guards ``b.mantissa != 0``.

    Underlies fixed-point ``//`` and ``%`` (their quotient is the same floor).
    The denominator's sign is normalised so Python's floor-toward-minus-infinity
    matches the true value's sign (keeps the // and % sign rules, cf. floating-point).
    """
    num = a.mantissa * 10**b.decimals
    den = b.mantissa * 10**a.decimals
    if den < 0:
        num, den = -num, -den
    return num // den


def _fp_value(num: int, den: int, a: FixedPoint, b: FixedPoint, exact_in: bool) -> "Value":
    """Quantize the exact result ``num/den`` to a FIXED_POINT Value.

    The result scale is ``max(a.decimals, b.decimals)`` — the precision that
    covers both operands (19.1.2). exactness propagates only if the inputs were
    exact AND the quantization lost nothing (an add/sub never loses; a mul/div
    that does not fit the scale rounds and is flagged inexact).

    When this quantization DOES round, the exact signed residual stored - true is
    carried on the Value's ``error`` (34.5.2) — bounded by half a ULP, the "how
    inexact" the analyze tree shows. A lossless quantization carries no error.
    """
    scale = max(a.decimals, b.decimals)
    mantissa, lossless = _fp_quantize(num, den, scale)
    error = None if lossless else Fraction(mantissa, 10**scale) - Fraction(num, den)
    return Value(
        Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=exact_in and lossless, error=error
    )


# --- internal high-precision pi (28.10.1) -------------------------------------
# An ENGINE primitive, NOT the abacus language's (still-unplanned) nullary pi()
# constant: sin/cos/... range-reduce their fixed-point argument mod 2*pi and need
# pi to the working scale to do it. Pure-stdlib integer arithmetic on scaled ints
# — no new dependency. Factored here so the trig family (cos/tan/cot, asin/...)
# all share the one implementation.

_PI_GUARD = 12  # internal extra digits so the returned scale's last place is right


def _arctan_inv(x: int, unity: int) -> int:
    """``arctan(1/x)`` scaled by ``unity`` (== 10**scale), as an integer.

    The Gregory/Leibniz series ``1/x - 1/(3x**3) + 1/(5x**5) - ...`` summed on
    scaled ints. ``power`` holds ``unity / x**(2k+1)``: start at ``unity // x``,
    then divide by ``x*x`` each step, contributing ``power // (2k+1)`` with an
    alternating sign. ``x`` is a small POSITIVE int (5, 239 for Machin), so every
    ``power`` is non-negative and the loop ends cleanly when it floors to 0.
    """
    x2 = x * x
    power = unity // x
    total = power
    k = 1
    while True:
        power //= x2
        if power == 0:
            break
        term = power // (2 * k + 1)
        total += -term if k % 2 else term
        k += 1
    return total


def _pi_scaled(decimals: int) -> int:
    """``floor(pi * 10**decimals)`` via Machin's formula, correct to the last place.

    ``pi == 16*arctan(1/5) - 4*arctan(1/239)``. Summed at ``decimals + _PI_GUARD``
    guard digits, then truncated back so the returned value is accurate to its own
    last digit (the guard absorbs the per-term truncation of the two series).
    """
    unity = 10 ** (decimals + _PI_GUARD)
    pi_guarded = 16 * _arctan_inv(5, unity) - 4 * _arctan_inv(239, unity)
    return pi_guarded // 10**_PI_GUARD


_E_GUARD = 12  # as _PI_GUARD: extra digits so the returned scale's last place is right


def _e_scaled(decimals: int) -> int:
    """``floor(e * 10**decimals)``, correct to the last place (29.3).

    ``e == exp(1) == sum 1/k!``, summed by the shared exp series at ``decimals +
    _E_GUARD`` guard digits then truncated back — the constant counterpart of
    ``_pi_scaled`` for the nullary ``e()``. Here ``z == unity`` (the argument 1.0),
    the one spot exp is summed without the callers' ln(2) range reduction; at z = 1
    the all-plus series still converges in a couple dozen terms and the guard
    absorbs their per-term truncation, the same way it does for pi.
    """
    unity = 10 ** (decimals + _E_GUARD)
    return _fp_exp_series(unity, unity) // 10**_E_GUARD  # exp(1) * unity, truncated back


def _fp_reduce_mod_2pi(fp: FixedPoint) -> tuple[int, int]:
    """Range-reduce a fixed-point radian argument into (-pi, pi] (28.10).

    Returns ``(reduced, working)``: ``reduced`` is the argument re-expressed as a
    scaled int at ``working`` decimals and shifted by a whole multiple of 2*pi
    into (-pi, pi], the small interval where a Taylor series around 0 converges
    fast and stays accurate. ``working`` is the operand's scale plus the argument's
    integer-digit count plus guard digits: the reduction subtracts up to
    ~10**int_digits multiples of 2*pi, so pi must carry that many extra places to
    keep the reduced argument good to ~1 ulp at ``decimals + guard``. Shared by the
    trig family (sin/cos/...), each of which then runs its own series on the result.
    """
    int_digits = len(str(abs(fp.mantissa) // 10**fp.decimals))
    working = fp.decimals + int_digits + _PI_GUARD
    pi = _pi_scaled(working)
    arg = fp.mantissa * 10 ** (working - fp.decimals)  # exact rescale to working scale
    reduced = arg % (2 * pi)
    if reduced > pi:
        reduced -= 2 * pi  # now in (-pi, pi]
    return reduced, working


def _fp_sin_series(reduced: int, unity: int) -> int:
    """Maclaurin sine of ``reduced/unity`` radians, as a scaled int at ``unity``.

    ``reduced`` must be NON-NEGATIVE: the integer term recurrence floors toward
    -inf, so a negative argument would stick the alternating series instead of
    converging to 0. Callers fold the sign off first (sin is odd) and restore it
    on the result. The Taylor sum ``x - x**3/3! + x**5/5! - ...`` runs on scaled
    ints: ``term_0 = reduced``; ``term_k = term_{k-1} * r**2 / ((2k)(2k+1))`` with
    ``r == reduced/unity``. Shared by sin (28.10) and tan (28.12).
    """
    reduced_sq = reduced * reduced
    term = reduced
    total = reduced
    k = 1
    while term != 0:
        term = term * reduced_sq // (unity * unity * (2 * k) * (2 * k + 1))
        total += -term if k % 2 else term
        k += 1
    return total


def _fp_cos_series(reduced: int, unity: int) -> int:
    """Maclaurin cosine of ``reduced/unity`` radians, as a scaled int at ``unity``.

    cos is EVEN, so callers pass ``abs(reduced)`` and there is no sign to restore
    (the non-negative argument also keeps the integer recurrence terminating). The
    Taylor sum ``1 - x**2/2! + x**4/4! - ...`` runs on scaled ints: ``term_0 = 1``
    (== ``unity``); ``term_k = term_{k-1} * r**2 / ((2k-1)(2k))`` with
    ``r == reduced/unity``. Shared by cos (28.11) and tan (28.12).
    """
    reduced_sq = reduced * reduced
    term = unity
    total = unity
    k = 1
    while term != 0:
        term = term * reduced_sq // (unity * unity * (2 * k - 1) * (2 * k))
        total += -term if k % 2 else term
        k += 1
    return total


# --- internal fixed-point inverse trig (28.14) --------------------------------
# The trig family's inverses, built on the SAME pi primitives above: where sin/cos
# range-reduce mod 2*pi and sum a forward Taylor series, asin reduces to a general-
# argument arctan series — the Machin-type arctan behind _pi_scaled (28.10.1) lifted
# from its unit-fraction _arctan_inv to an arbitrary scaled argument. ENGINE
# primitives shared by the inverse-trig family (asin/acos/atan, 28.14-28.16).


def _fp_arctan_series(z: int, unity: int) -> int:
    """``arctan(z/unity)`` scaled by ``unity``, for ``0 <= z <= unity``, as an int.

    The general-argument, alternating sibling of ``_fp_atanh_series`` (all-plus) and
    of ``_arctan_inv`` (unit-fraction argument). The bare Gregory series
    ``z - z**3/3 + z**5/5 - ...`` converges slowly as the argument nears 1, so ONE
    half-angle reduction ``arctan(z) == 2*arctan(z / (1 + sqrt(1 + z**2)))`` first
    pulls any argument in ``[0, 1]`` down to ``<= sqrt(2)-1`` (~0.414) for fast
    convergence, and the result is doubled back. ``z`` must be NON-NEGATIVE (the
    integer recurrence floors toward -inf); arctan is odd, so callers fold the sign
    off first. Shared by the inverse-trig family (asin/acos/atan, 28.14-28.16).
    """
    root = math.isqrt((unity + z * z // unity) * unity)  # sqrt(1 + (z/unity)**2) * unity
    z = z * unity // (unity + root)  # the half-angle argument, now <= ~0.414 * unity
    z_sq = z * z
    term = z
    total = z
    k = 1
    while term != 0:
        term = term * z_sq // (unity * unity)
        total += -(term // (2 * k + 1)) if k % 2 else term // (2 * k + 1)
        k += 1
    return 2 * total


def _fp_asin_scaled(mantissa: int, decimals: int) -> tuple[int, int]:
    """Arcsine of ``x == mantissa/10**decimals`` for ``0 <= mantissa <= 10**decimals``
    (so ``0 <= x <= 1``), as ``(asin_scaled, working)`` at ``working == decimals +
    _PI_GUARD`` (28.14). The caller quantizes back (asin) or offsets it by pi/2
    before rounding (acos, 28.15) — returning the un-rounded working-scale value lets
    acos round only once. NON-NEGATIVE argument only: asin is odd, so callers fold
    the sign off and restore it on the result.

    ``asin(x) == atan(x / sqrt(1 - x**2))`` — the plain arcsine series converges
    badly near ``x == 1``, so route through arctan instead, reusing the integer
    sqrt and the arctan series. The arctan argument ``u == x/sqrt(1-x**2)`` exceeds
    1 once ``x > 1/sqrt(2)``, beyond the series' domain, so reduce it there with
    ``atan(u) == pi/2 - atan(1/u)`` (and ``1/u == sqrt(1-x**2)/x <= 1``).
    """
    working = decimals + _PI_GUARD
    unity = 10**working
    x = mantissa * 10 ** (working - decimals)  # x scaled, in [0, unity]
    root = math.isqrt((unity - x * x // unity) * unity)  # sqrt(1 - x**2) scaled, in [0, unity]
    if x <= root:  # x <= 1/sqrt(2): the arctan argument u = x/sqrt(1-x**2) is <= 1
        atan = _fp_arctan_series(x * unity // root, unity)
    else:  # u > 1 (root may be 0 at x = 1): asin(x) = pi/2 - arctan(1/u)
        atan = _pi_scaled(working) // 2 - _fp_arctan_series(root * unity // x, unity)
    return atan, working


def _fp_asin(mantissa: int, decimals: int) -> int:
    """Arcsine of a NON-NEGATIVE ``x == mantissa/10**decimals`` (``0 <= x <= 1``) as a
    scaled-int mantissa at ``decimals`` (28.14) — the ``_fp_asin_scaled`` core rounded
    half-to-even back to the operand's scale. The caller folds the sign (asin is odd).
    """
    atan, working = _fp_asin_scaled(mantissa, decimals)
    out, _ = _fp_quantize(atan, 10 ** (working - decimals), 0)
    return out


def _fp_atan(mantissa: int, decimals: int) -> int:
    """Arctangent of a NON-NEGATIVE ``x == mantissa/10**decimals`` (``x >= 0``, NO
    upper bound — atan's domain is all reals) as a scaled-int mantissa at ``decimals``
    (28.16). The caller folds the sign (atan is odd).

    The arctan series (``_fp_arctan_series``) only converges on arguments in [0, 1],
    so arguments above 1 reduce with ``atan(x) == pi/2 - atan(1/x)`` (and
    ``1/x < 1``), the same identity asin uses for its over-unit arctan argument.
    Computed at ``decimals + _PI_GUARD`` guard digits, then rounded half-to-even back.
    """
    working = decimals + _PI_GUARD
    unity = 10**working
    x = mantissa * 10 ** (working - decimals)  # x scaled, in [0, inf)
    if x <= unity:  # x <= 1: the series argument is in range
        atan = _fp_arctan_series(x, unity)
    else:  # x > 1: atan(x) = pi/2 - atan(1/x), with 1/x = unity/x < 1
        atan = _pi_scaled(working) // 2 - _fp_arctan_series(unity * unity // x, unity)
    out, _ = _fp_quantize(atan, 10 ** (working - decimals), 0)
    return out


# --- internal high-precision natural log (28.17) ------------------------------
# The log family's analogue of the trig family above: where sin/cos range-reduce
# mod 2*pi and sum a Taylor series, log range-reduces in base 10 and sums an
# atanh series. ENGINE primitives, NOT the abacus language's log() — that is the
# thin two-step add (Value.log) which calls _fp_ln. Pure-stdlib integer arithmetic
# on scaled ints, no new dependency, and factored here so log10/log2 (28.18/28.19)
# reuse the one ln core plus the ln(10)/ln(2) constants.

_LN_GUARD = 12  # internal extra digits so the returned scale's last place is right


def _fp_atanh_series(t: int, unity: int) -> int:
    """``atanh(t/unity)`` scaled by ``unity``, as an integer (28.17.1).

    The all-plus odd series ``t + t**3/3 + t**5/5 + ...`` summed on scaled ints —
    the log-family analogue of ``_fp_sin_series`` and the all-plus sibling of the
    alternating ``_arctan_inv``. ``term_k = term_{k-1} * t**2 / unity**2`` carries
    ``unity * (t/unity)**(2k+1)``; each contributes ``term // (2k+1)``. ``t`` must
    be NON-NEGATIVE: the integer recurrence floors toward -inf, so a negative
    argument would stick the series instead of converging to 0. atanh is odd, so
    callers fold the sign off first and restore it on the result. Shared by the ln
    core (general ``t``) and the ln constants (``t = unity // k``, a unit fraction).
    """
    t_sq = t * t
    term = t
    total = t
    k = 1
    while term != 0:
        term = term * t_sq // (unity * unity)
        total += term // (2 * k + 1)
        k += 1
    return total


def _ln10_scaled(decimals: int) -> int:
    """``floor(ln(10) * 10**decimals)``, correct to the last place (28.17.2).

    ``ln(10) == 6*atanh(1/3) + 2*atanh(1/9)`` — since ``2*atanh(1/3) == ln(2)`` and
    ``2*atanh(1/9) == ln(5/4)``, the sum is ``ln(8) + ln(5/4) == ln(10)``. The
    log-family analogue of ``_pi_scaled``'s Machin formula; summed at ``decimals +
    _LN_GUARD`` guard digits then truncated back so the last place is right.
    """
    unity = 10 ** (decimals + _LN_GUARD)
    ln10_guarded = 6 * _fp_atanh_series(unity // 3, unity) + 2 * _fp_atanh_series(unity // 9, unity)
    return ln10_guarded // 10**_LN_GUARD


def _ln2_scaled(decimals: int) -> int:
    """``floor(ln(2) * 10**decimals)``, correct to the last place (28.17.2).

    ``ln(2) == 2*atanh(1/3)`` (since ``(1+1/3)/(1-1/3) == 2``). The base-2 constant
    behind log2 (28.19); the same guard-then-truncate shape as ``_ln10_scaled``.
    """
    unity = 10 ** (decimals + _LN_GUARD)
    ln2_guarded = 2 * _fp_atanh_series(unity // 3, unity)
    return ln2_guarded // 10**_LN_GUARD


def _power_of_ten_exponent(n: int) -> int | None:
    """The integer ``j`` with ``n == 10**j``, or None if ``n`` is not a power of ten.

    ``n`` must be a positive integer. log10 (28.18) uses it to spot its exact
    landmark — a power of ten logs to a whole number with no rounding, so the
    result is that exponent verbatim.
    """
    j = len(str(n)) - 1
    return j if 10**j == n else None


def _integer_log(x: Fraction, base: Fraction) -> int | None:
    """The integer ``k`` with ``base**k == x`` exactly, or None (40.10).

    The exact landmark of the two-arg ``log(x, base)``: when ``x`` is a whole power
    of the base the result is that exponent with no rounding — ``log(8, 2) == 3``,
    ``log(1, base) == 0``, ``log(1/4, 2) == -2``. ``x`` and ``base`` are POSITIVE
    rationals and ``base != 1`` (the caller guards the domain). A float estimate of
    ``log_base(x)`` narrows the search to a few candidate exponents, each VERIFIED
    by exact ``Fraction`` exponentiation, so a near-miss is rejected — only a true
    power returns. A candidate exponent past ``_INTEGER_LOG_LIMIT`` is skipped: its
    exact power would be astronomically large and is never a landmark in practice.
    """
    if x == 1:
        return 0  # base**0 == 1 for every base
    try:
        estimate = math.log(float(x)) / math.log(float(base))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    for k in {math.floor(estimate), math.ceil(estimate), round(estimate)}:
        if abs(k) <= _INTEGER_LOG_LIMIT and base**k == x:
            return k
    return None


_INTEGER_LOG_LIMIT = 4096  # cap on the candidate exponent _integer_log will verify

_MAX_FACTORIAL = 1000  # cap on n in factorial(n); keeps a huge operand from blowing up


def _fp_ln(fp: FixedPoint) -> tuple[int, int]:
    """Natural log of a POSITIVE fixed-point value (28.17), as a scaled int.

    Returns ``(ln_total, working)`` where ``ln_total / 10**working`` approximates
    ``ln(x)``; the caller quantizes back to its own scale (and log10/log2 first
    divide by the ln(10)/ln(2) constant at ``working``). The caller guards
    ``mantissa > 0`` — ln of a non-positive value is undefined.

    Base-10 argument reduction (EXACT here — a bare mantissa shift): write
    ``x == r * 10**n`` with ``n == round(log10(x))`` so ``r`` lands in
    ``[sqrt(0.1), sqrt(10))``, then ``ln(x) == n*ln(10) + ln(r)`` and ``ln(r) ==
    2*atanh((r-1)/(r+1))`` with ``|t| <= 0.52`` for fast convergence. ``n`` falls
    out of the digit count: ``f == floor(log10(x)) == len(str(mantissa)) - 1 -
    decimals``, bumped to ``f + 1`` when ``x >= sqrt(10)*10**f`` (tested as
    ``mantissa**2 >= 10**(2*decimals + 2*f + 1)``, exact integer arithmetic).
    """
    m, d = fp.mantissa, fp.decimals
    f = len(str(m)) - 1 - d  # floor(log10(x))
    n = f + 1 if m * m >= 10 ** (2 * d + 2 * f + 1) else f
    working = d + _LN_GUARD + max(n, 0)  # keep the reduced argument exact (exp >= guard)
    unity = 10**working
    reduced = m * 10 ** (working - d - n)  # r * unity, exact; r in [sqrt(.1), sqrt(10))
    t = (reduced - unity) * unity // (reduced + unity)  # ((r-1)/(r+1)) * unity
    sign = -1 if t < 0 else 1  # atanh is odd; sum the series on |t|
    ln_r = 2 * sign * _fp_atanh_series(abs(t), unity)
    return n * _ln10_scaled(working) + ln_r, working


# --- internal inverse hyperbolics (40.3) --------------------------------------
# Each reduces to the natural-log core (_fp_ln, 28.17) of a sqrt-built argument,
# computed at _LN_GUARD guard digits and quantized half-to-even back to the
# operand's scale. The forward hyperbolics (40.2) go the other way, through exp.
# asinh/atanh are odd, so the Value methods fold the sign and these take |x|.


def _fp_asinh(mantissa: int, decimals: int) -> int:
    """asinh(|x|) for ``x == mantissa/10**decimals``, as a scaled int at ``decimals``
    (40.3). ``asinh(x) == ln(x + sqrt(x**2 + 1))``; the caller folds the sign (asinh is
    odd), so the NON-NEGATIVE argument keeps ``x + sqrt(...) >= 1`` — no cancellation
    (for a negative x the two terms would nearly cancel)."""
    w = decimals + _LN_GUARD
    unity = 10**w
    x = abs(mantissa) * 10 ** (w - decimals)  # |x| * unity, exact
    root = math.isqrt(x * x + unity * unity)  # floor(sqrt(x**2 + 1) * unity)
    ln_total, working = _fp_ln(FixedPoint(x + root, w))  # arg >= unity, so ln >= 0
    out, _ = _fp_quantize(ln_total, 10 ** (working - decimals), 0)
    return out


def _fp_acosh(mantissa: int, decimals: int) -> int:
    """acosh(x) for ``x == mantissa/10**decimals`` with ``x > 1`` (the caller handles
    the ``x == 1`` landmark and refuses ``x < 1``), as a scaled int at ``decimals``
    (40.3). ``acosh(x) == ln(x + sqrt(x**2 - 1))``; ``x > 1`` keeps the radicand
    positive and the argument above 1."""
    w = decimals + _LN_GUARD
    unity = 10**w
    x = mantissa * 10 ** (w - decimals)  # x * unity, x > 1
    root = math.isqrt(x * x - unity * unity)  # floor(sqrt(x**2 - 1) * unity)
    ln_total, working = _fp_ln(FixedPoint(x + root, w))
    out, _ = _fp_quantize(ln_total, 10 ** (working - decimals), 0)
    return out


def _fp_atanh(mantissa: int, decimals: int) -> int:
    """atanh(|x|) for ``x == mantissa/10**decimals`` with ``|x| < 1`` (the caller folds
    the sign and refuses ``|x| >= 1``), as a scaled int at ``decimals`` (40.3).
    ``atanh(x) == ln((1 + x)/(1 - x))/2`` — through the PUBLIC ln reduction, NOT the bare
    ``_fp_atanh_series`` (which is ln's own core); the ``/2`` folds into the quantize."""
    w = decimals + _LN_GUARD
    unity = 10**w
    x = abs(mantissa) * 10 ** (w - decimals)  # |x| * unity, 0 < x < unity
    arg = (unity + x) * unity // (unity - x)  # floor(((1 + |x|)/(1 - |x|)) * unity) >= unity
    ln_total, working = _fp_ln(FixedPoint(arg, w))
    out, _ = _fp_quantize(ln_total, 2 * 10 ** (working - decimals), 0)  # ln(arg)/2
    return out


def _fp_exp_series(z: int, unity: int) -> int:
    """``exp(z/unity)`` scaled by ``unity``, as an integer (28.20.1, Path B).

    The all-plus series ``1 + z + z**2/2! + z**3/3! + ...`` summed on scaled ints,
    the exp half of the log-family pair (the inverse of ``_fp_ln``). ``z`` must be
    NON-NEGATIVE and SMALL — callers range-reduce by ln(2) so ``z/unity`` lands in
    ``[0, ln 2)`` (fast convergence) and restore the ``2**k`` factor afterwards; a
    negative argument would stick the integer recurrence (it floors toward -inf).
    ``term_k = term_{k-1} * (z/unity) / k`` carries ``unity * (z/unity)**k / k!``.
    """
    term = unity
    total = unity
    k = 1
    while term != 0:
        term = term * z // (unity * k)
        total += term
        k += 1
    return total


def _fp_exp_ratio(fp: "FixedPoint") -> Fraction:
    """``exp(x)`` for ``x == fp`` as an exact Fraction, UN-rounded (28.27 / 40.2).

    The shared exp core: the all-plus Taylor series (``_fp_exp_series``) range-reduced
    by ln(2) into ``2**k * exp(s)`` with ``s in [0, ln 2)`` — the ``2**k`` an exact
    mantissa shift, only ``exp(s)`` summed. The working scale carries the result's
    ``~k*log10(2)`` integer digits on top of the operand's so the shift stays accurate
    to the last place; negative ``x`` is fine (``k < 0``). Returned as the raw rational
    so a caller can quantize ONCE — ``exp`` (28.27) rounds it straight back, the
    hyperbolics (40.2) combine ``exp(x)`` with ``exp(-x)`` before a single rounding.
    """
    d = fp.decimals
    # exp(x) ~ 2**k grows with x, so the working scale must cover the result's integer
    # digits; size it from a cheap first estimate of k (2**(k+1) bounds exp(x),
    # len(str(...)) is its exact digit count).
    coarse = d + _LN_GUARD
    k_est = fp.mantissa * 10 ** (coarse - d) // _ln2_scaled(coarse)
    headroom = len(str(1 << (k_est + 1))) if k_est > 0 else 0
    working = d + _LN_GUARD + headroom
    unity = 10**working
    ln2 = _ln2_scaled(working)
    x = fp.mantissa * 10 ** (working - d)  # x at the working scale, exact
    k = x // ln2  # floor keeps s in [0, ln 2), where _fp_exp_series needs z >= 0
    s = x - k * ln2
    es = _fp_exp_series(s, unity)  # exp(s) * unity
    num, den = (es << k, unity) if k >= 0 else (es, unity << -k)
    return Fraction(num, den)


def _fp_exp_pm(fp: "FixedPoint") -> tuple[Fraction, Fraction]:
    """``(exp(x), exp(-x))`` as exact Fractions — the hyperbolics' two exponentials (40.2).

    sinh/cosh/tanh all combine ``e**x`` and ``e**-x``; computing both un-rounded here
    lets each combine then quantize once (no double rounding). The caller has already
    handled the ``x == 0`` landmark, so neither exponential is the trivial 1.
    """
    return _fp_exp_ratio(fp), _fp_exp_ratio(FixedPoint(-fp.mantissa, fp.decimals))


def _iroot(n: int, q: int) -> int:
    """``floor(n ** (1/q))`` for ``n >= 0``, ``q >= 1`` — integer q-th root (28.20.1).

    Integer Newton from a bit-length over-estimate, then a one-step correction so
    the result is the exact floor regardless of any rounding drift. The bignum
    generalization of ``math.isqrt`` (which only does q = 2); pow's Path A uses it
    to test whether a q-th root lands exactly on the grid.
    """
    if n in (0, 1) or q == 1:
        return n
    x = 1 << -(-n.bit_length() // q)  # 2**ceil(bits/q) >= n**(1/q)
    while True:
        t = ((q - 1) * x + n // x ** (q - 1)) // q
        if t >= x:
            break
        x = t
    while x**q > n:
        x -= 1
    while (x + 1) ** q <= n:
        x += 1
    return x


def _perfect_root(n: int, q: int) -> int | None:
    """The exact q-th root of ``n >= 0`` if ``n`` is a perfect q-th power, else None.

    The generalization of sqrt's perfect-square test (a q-th root is rational iff
    the radicand is a perfect q-th power); pow's Path A applies it to a fraction's
    numerator and denominator independently.
    """
    r = _iroot(n, q)
    return r if r**q == n else None


# --- bitwise helpers (24.3.2) -------------------------------------------------
# Bitwise ops act on each mode's OWN stored bits (the decision: offer them on
# every type, not integer-only, and let the caller judge what a bit manipulation
# means). Operand sign/width then follow Python's native int semantics — for the
# common non-negative case that IS "wider operand wins, narrower zero-extended";
# negatives sign-extend (the C/Python convention this is a tool for, 24.3).

_FLOAT64_MASK = (1 << 64) - 1  # the 64 bits an IEEE-754 double occupies


def _float_bits(x: float) -> int:
    """A double's raw IEEE-754 pattern as an unsigned 64-bit int (cf. details())."""
    return int.from_bytes(struct.pack(">d", x), "big")


def _bits_float(bits: int) -> float:
    """Reinterpret the low 64 bits of ``bits`` as a double (inverse of _float_bits).

    Masked to 64 bits so a ``~`` result (a negative Python int with infinite
    leading ones) maps back to exactly the flipped 64-bit pattern.
    """
    return struct.unpack(">d", (bits & _FLOAT64_MASK).to_bytes(8, "big"))[0]


def _fp_align(a: FixedPoint, b: FixedPoint) -> tuple[int, int, int]:
    """Both mantissas re-expressed at the common scale ``max(decimals)`` (24.3.2).

    Returns ``(ma, mb, scale)``. Padding a mantissa to a larger scale multiplies
    by a power of ten and is exact, so the operands' bits are compared at the same
    decimal alignment (1.5 & 3 == 15 & 30, not 15 & 3) and the result carries that
    covering scale — the same max()-of-scales every fixed-point binary op uses.
    """
    scale = max(a.decimals, b.decimals)
    return a.mantissa * 10 ** (scale - a.decimals), b.mantissa * 10 ** (scale - b.decimals), scale


def _lexeme_scale(lexeme: str) -> int:
    """The written decimal scale of a numeric literal, from its verbatim text (29.3).

    The fractional-digit count the lexeme spells out: ``D`` for an M@D literal,
    the digits after the point for a decimal/scientific form (0 once a positive
    exponent absorbs them), 0 for a plain or base-prefixed integer. IDENTICAL to
    the ``decimals`` ``from_lexeme`` assigns a FIXED_POINT value before the floor —
    read straight off the source so the nullary-precision pre-walk (29.3) need not
    evaluate the tree. Mirrors from_lexeme's three literal cases.
    """
    if lexeme[-1:] in ("i", "j"):  # imaginary suffix (complex mode): scale of its magnitude
        lexeme = lexeme[:-1]
    if "@" in lexeme:  # M@D (20.5): the scale is the explicit '@<decimals>' tail
        return int(lexeme.partition("@")[2])
    if lexeme[:2].lower() in {"0x", "0b", "0o"}:  # base-prefixed integers (20.4)
        return 0
    exponent = Decimal(lexeme).as_tuple()[2]  # exact, unrounded — as from_lexeme
    assert isinstance(exponent, int)  # only 'n'/'F' for nan/inf, never lexed
    return -exponent if exponent < 0 else 0  # positive exponent is an integer, scale 0


@dataclass(frozen=True, slots=True)
class Value:
    """A number in ONE mode's own representation, plus an exactness flag.

    The ONE class that can hold a value in ANY supported mode (10.1.x).
    Conceptually ``(mode, payload, exact)``: ``mode`` is the numeric regime and
    its parameters (precision / scale / rounding / overflow rule, 10.2.x);
    ``payload`` is that mode's OWN representation (fixed-point (mantissa,
    decimals), floating-point float, Decimal, Fraction, masked fixed-width int, ...);
    ``exact`` is whether this value exactly equals the number it stands for.

    Storage is wired for FLOATING_POINT (19.1.1), FIXED_POINT (19.1.2) and RATIONAL
    (19.1.7); the ``payload`` union widens by one type as each further member of
    ``Mode`` (19.1.3-19.1.6, 19.1.8) is built.
    """

    mode: Mode
    # per-mode storage: FLOATING_POINT float, FIXED_POINT FixedPoint, RATIONAL Fraction,
    # COMPLEX Complex (a pair of FixedPoints), VECTOR Vector (a tuple of element Values)
    payload: Fraction | float | FixedPoint | Complex | Vector
    exact: bool
    # "How inexact" (34.5.2): the EXACT signed quantization residual stored - true
    # that THIS value's own rounding introduced, as a Fraction — None when nothing
    # rounded here (an exact value) or when the residual is not a clean rational
    # (an irrational-root / transcendental rounding, where the "true" value is the
    # series approximation, not the real number). Set only at the fixed-point
    # algebraic chokepoint (_fp_value). Pure annotation: excluded from equality and
    # hashing so a value's identity stays (mode, payload, exact).
    #
    # NOTE the residual is a Fraction, NOT a fixed-point number, computed in exact
    # rational arithmetic (Fraction(mantissa, 10**scale) - Fraction(num, den)). It
    # has to be: the true value it measures the distance to (e.g. 100/3) is often
    # not representable in fixed-point at any scale, so the error isn't either.
    # Expressing it in fixed-point would round it — "the error of the rounding,
    # rounded" — defeating the field. So the CALCULATION stays fixed-point; this
    # DIAGNOSTIC about it borrows rational, the engine's exact ground-truth type.
    error: Fraction | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.error is not None and type(self.error) is not Fraction:
            raise ValueError("Value error must be a Fraction or None")
        # Per-mode storage validation — one case per type, grows with the enum.
        match self.mode:
            case Mode.FLOATING_POINT:
                if type(self.payload) is not float:
                    raise ValueError("FLOATING_POINT payload must be a float")
            case Mode.FIXED_POINT:
                if type(self.payload) is not FixedPoint:
                    raise ValueError("FIXED_POINT payload must be a FixedPoint")
            case Mode.RATIONAL:
                if type(self.payload) is not Fraction:
                    raise ValueError("RATIONAL payload must be a Fraction")
            case Mode.COMPLEX:
                if type(self.payload) is not Complex:
                    raise ValueError("COMPLEX payload must be a Complex")
            case Mode.VECTOR:
                if type(self.payload) is not Vector:
                    raise ValueError("VECTOR payload must be a Vector")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    @classmethod
    def from_lexeme(cls, lexeme: str, mode: "Mode", min_decimals: int = 0) -> "Value":
        """Interpret the raw source lexeme in the mode's representation (19.2).

        Handles every 20.1.3 form: integers, decimals, scientific, the
        base-prefixed integers 0xFF / 0b1010 / 0o17 (20.4), and the M@D notation
        <base-int>@<decimals> (20.5, == M * 10**-D; '@' attaches only to
        base-prefixed integers, never a decimal — 19.2.1). A base-prefixed or M@D lexeme
        is mode-agnostic — parsed once, then stored in each mode's own
        representation. Sign is NOT here: it is a unary op on the parsed Value
        (17.2.2), never part of the lexeme.

        ``min_decimals`` is the caller's min_fixed_point_precision floor (25.2.1):
        a fixed-point literal is held at a scale of at least this many fractional
        digits, so the max()-of-operand-scales propagation carries the floor
        through the whole calculation and a `/` that would round at scale 0 keeps
        more decimals instead. It only affects FIXED_POINT (other modes have no
        decimal scale) and defaults to 0 — no floor, behaviour unchanged.
        """
        # An imaginary literal (trailing 'i'/'j', lexed only onto a decimal) is only
        # meaningful in complex mode; refuse it cleanly elsewhere rather than letting
        # the per-mode parse choke on the suffix.
        if lexeme[-1:] in ("i", "j") and mode is not Mode.COMPLEX:
            raise NotRepresentableError("imaginary literal requires complex mode")
        # M@D (20.5): an integer mantissa '@' a decimal scale, == M * 10**-D.
        # Like the base-prefixed integers it is valid in EVERY mode (the
        # side-by-side product needs the same literal to evaluate in all modes).
        if "@" in lexeme:
            mant_str, _, dec_str = lexeme.partition("@")
            return cls._from_scaled_int(int(mant_str, 0), int(dec_str), mode, min_decimals)
        # 0x/0b/0o lexemes are integers; float() rejects them, so parse first.
        base_prefixed = lexeme[:2].lower() in {"0x", "0b", "0o"}
        match mode:
            case Mode.FLOATING_POINT:
                # floating-point carries exact=False unconditionally (provisional —
                # 1.0 + 2.0 IS exact; refine when the exactness rule settles).
                number = float(int(lexeme, 0)) if base_prefixed else float(lexeme)
                return cls(Mode.FLOATING_POINT, number, exact=False)
            case Mode.FIXED_POINT:
                if base_prefixed:
                    return cls._from_scaled_int(int(lexeme, 0), 0, mode, min_decimals)
                # Decimal parses every decimal/scientific form EXACTLY (the
                # constructor is not subject to context rounding); as_tuple gives
                # the scaled integer M * 10**exp that maps straight to (M, -exp).
                sign, digits, exponent = Decimal(lexeme).as_tuple()
                assert isinstance(exponent, int)  # only 'n'/'F' for nan/inf, never lexed
                n = -int("".join(map(str, digits))) if sign else int("".join(map(str, digits)))
                if exponent >= 0:
                    return cls._from_scaled_int(n * 10**exponent, 0, mode, min_decimals)
                return cls._from_scaled_int(n, -exponent, mode, min_decimals)
            case Mode.RATIONAL:
                # rational holds the literal EXACTLY: 0.1 -> 1/10, not a float.
                # Every literal is exact, so rational values start exact=True.
                fraction = Fraction(int(lexeme, 0)) if base_prefixed else Fraction(lexeme)
                return cls(Mode.RATIONAL, fraction, exact=True)
            case Mode.COMPLEX:
                # A trailing 'i'/'j' (lexed only onto a decimal) makes the literal
                # IMAGINARY (real part 0); otherwise it is a real (imag part 0). Either
                # way the magnitude parses as a FIXED_POINT part, so a complex literal
                # inherits the fixed-point scale of its digits.
                imaginary = lexeme[-1:] in ("i", "j")
                body = lexeme[:-1] if imaginary else lexeme
                part = cls.from_lexeme(body, Mode.FIXED_POINT, min_decimals)
                assert isinstance(part.payload, FixedPoint)
                zero = FixedPoint(0, part.payload.decimals)
                parts = Complex(zero, part.payload) if imaginary else Complex(part.payload, zero)
                return cls(Mode.COMPLEX, parts, exact=part.exact)
            case _:
                raise ValueError(f"unsupported mode: {mode!r}")

    @classmethod
    def _from_scaled_int(
        cls, mantissa: int, decimals: int, mode: "Mode", min_decimals: int = 0
    ) -> "Value":
        """Build a Value from the exact decimal ``mantissa * 10**-decimals``.

        Shared constructor for the M@D notation (every mode) and FIXED_POINT's
        own decimal literals. Each mode stores the same exact value in its
        representation; floating-point rounds to the nearest double (correctly, via
        Fraction) and is the only inexact one.

        ``min_decimals`` (25.2.1) raises the FIXED_POINT scale to at least that
        many fractional digits — the same value, re-scaled by padding zeros, so
        it stays exact; it is the one place the min_fixed_point_precision floor is
        applied. The floor is a minimum: a literal already wider than it keeps its
        own scale (max). Ignored by modes without a decimal scale.
        """
        match mode:
            case Mode.FLOATING_POINT:
                number = float(Fraction(mantissa, 10**decimals))
                return cls(Mode.FLOATING_POINT, number, exact=False)
            case Mode.FIXED_POINT:
                scale = max(decimals, min_decimals)
                return cls(
                    Mode.FIXED_POINT,
                    FixedPoint(mantissa * 10 ** (scale - decimals), scale),
                    exact=True,
                )
            case Mode.RATIONAL:
                return cls(Mode.RATIONAL, Fraction(mantissa, 10**decimals), exact=True)
            case Mode.COMPLEX:
                real = cls._from_scaled_int(mantissa, decimals, Mode.FIXED_POINT, min_decimals)
                assert isinstance(real.payload, FixedPoint)
                return cls(
                    Mode.COMPLEX,
                    Complex(real.payload, FixedPoint(0, real.payload.decimals)),
                    exact=real.exact,
                )
            case _:
                raise ValueError(f"unsupported mode: {mode!r}")

    @classmethod
    def from_real(cls, x: "float | int | Fraction", mode: "Mode", scale: int) -> "Value":
        """Materialise an arbitrary real number into ``mode`` at ``scale`` decimals (31.7).

        The solver's bridge from a search candidate (a plain ``float`` the
        golden-section engine carries) back into a mode-faithful Value the program
        can be evaluated against. Unlike ``from_lexeme`` there is no source text —
        the input is a number already — so it routes through the SAME chokepoints
        the literal path uses (``_fp_quantize`` to round to the scale,
        ``_from_scaled_int`` to build the payload), keeping one place that decides
        how a real lands in each representation.

        - floating-point: the nearest double (``scale`` is irrelevant — a double has
          no decimal scale), inexact like every float here.
        - fixed-point / rational: the exact rational ``x`` rounded half-to-even to
          ``scale`` decimals, then stored exactly at that scale. The candidate IS
          that representable value exactly, so it carries ``exact=True``; any
          inexactness in a solver answer comes from the EXPRESSION's own rounding at
          this point, not from the binding.
        """
        frac = x if isinstance(x, Fraction) else Fraction(x)
        match mode:
            case Mode.FLOATING_POINT:
                return cls(Mode.FLOATING_POINT, float(frac), exact=False)
            case Mode.FIXED_POINT | Mode.RATIONAL:
                mantissa, _lossless = _fp_quantize(frac.numerator, frac.denominator, scale)
                return cls._from_scaled_int(mantissa, scale, mode)
            case Mode.COMPLEX:
                # The solver drives a REAL candidate; complex search is unsupported, so
                # there is no real-to-complex binding to materialise here.
                raise NotRepresentableError(
                    "complex mode has no real-candidate binding (the solver is real-valued)"
                )
            case _:
                raise ValueError(f"unsupported mode: {mode!r}")

    # --- nullary functions (29.2 / 29.3 / 28.1) -------------------------
    # Zero-argument functions like pi(), e() and time(). UNLIKE every other function
    # they are NOT operand-methods (no ``self`` operand carries the mode): each takes
    # the per-run EvalContext (29.1) and builds a Value in ``ctx.mode``. They are
    # the "registered-callable kind that takes the eval context" the nodes registry
    # dispatches to. The irrational CONSTANTS (pi/e) are inexact-or-refuse in every
    # mode: float rounds its native double, fixed-point truncates to the run's
    # derived scale (ctx.nullary_precision, 29.3), and rational has no finite scale
    # to round to so it refuses — the same exact-or-refuse stance as sqrt/sin. The
    # CLOCK nullary time() (28.1) breaks that pattern: a tick is exactly rational, so
    # it refuses in no mode and is inexact only in float (see its own docstring).

    @classmethod
    def pi(cls, ctx: "EvalContext") -> "Value":
        """The circle constant pi in ``ctx.mode`` (29.3).

        float: math.pi (the nearest double, inexact). fixed-point: pi truncated to
        ``ctx.nullary_precision`` decimals via the engine's Machin-formula helper,
        inexact (irrational). rational: refuses — pi is irrational with no scale.
        """
        match ctx.mode:
            case Mode.FLOATING_POINT:
                return cls(Mode.FLOATING_POINT, math.pi, exact=False)
            case Mode.FIXED_POINT:
                scale = ctx.nullary_precision
                return cls(Mode.FIXED_POINT, FixedPoint(_pi_scaled(scale), scale), exact=False)
            case Mode.RATIONAL:
                raise NotRepresentableError("pi is irrational; no rational value")
            case Mode.COMPLEX:
                scale = ctx.nullary_precision
                return cls(
                    Mode.COMPLEX,
                    Complex(FixedPoint(_pi_scaled(scale), scale), FixedPoint(0, scale)),
                    exact=False,
                )
            case _:
                raise ValueError(f"unsupported mode: {ctx.mode!r}")

    @classmethod
    def e(cls, ctx: "EvalContext") -> "Value":
        """Euler's number e in ``ctx.mode`` (29.3) — mirror of ``pi``.

        float: math.e. fixed-point: e truncated to ``ctx.nullary_precision``
        decimals via the exp-series helper, inexact. rational: refuses (irrational).
        """
        match ctx.mode:
            case Mode.FLOATING_POINT:
                return cls(Mode.FLOATING_POINT, math.e, exact=False)
            case Mode.FIXED_POINT:
                scale = ctx.nullary_precision
                return cls(Mode.FIXED_POINT, FixedPoint(_e_scaled(scale), scale), exact=False)
            case Mode.RATIONAL:
                raise NotRepresentableError("e is irrational; no rational value")
            case Mode.COMPLEX:
                scale = ctx.nullary_precision
                return cls(
                    Mode.COMPLEX,
                    Complex(FixedPoint(_e_scaled(scale), scale), FixedPoint(0, scale)),
                    exact=False,
                )
            case _:
                raise ValueError(f"unsupported mode: {ctx.mode!r}")

    @classmethod
    def time(cls, ctx: "EvalContext") -> "Value":
        """The current Unix epoch (UTC) in ``ctx.mode`` — the first clock nullary (28.1).

        Reads the ONE instant the run was sampled at (``ctx.now_ns``, integer
        nanoseconds since the epoch, 28.1.2), then renders it per mode through the
        ``_from_scaled_int`` chokepoint (28.1.1). UNLIKE pi/e a clock tick is
        EXACTLY rational, so this refuses in no mode and is inexact only where the
        type rounds:

        - fixed-point: at the run's derived scale s (``nullary_precision``, 29.3).
          s == 0 (the default, no literal) -> whole seconds, the C library
          ``time()``; 1..9 -> seconds + ``tv_nsec`` TRUNCATED to s decimals; >= 9
          -> the full ns reading (further decimals are true zeros — the clock has
          no finer grain). Exact: the rendered decimal IS the sampled value, the
          scale is a resolution choice, not a rounding.
        - float: the full ns as a native double (~sub-microsecond at epoch
          magnitude) -> the only mode that rounds, so inexact.
        - rational: the full ns as ``Fraction(now_ns, 10**9)`` -> exact, no refuse.
        """
        now_ns = ctx.now_ns
        if now_ns is None:  # defensive: evaluate() always samples before a time() run
            raise ValueError("time() needs a sampled clock reading (ctx.now_ns)")
        match ctx.mode:
            case Mode.FIXED_POINT:
                scale = ctx.nullary_precision
                # Truncate the ns reading to `scale` fractional digits (>= 9 pads
                # true zeros); the rendered decimal is exactly the sampled value.
                if scale <= 9:
                    mantissa = now_ns // 10 ** (9 - scale)
                else:
                    mantissa = now_ns * 10 ** (scale - 9)
                return cls._from_scaled_int(mantissa, scale, Mode.FIXED_POINT)
            case Mode.FLOATING_POINT | Mode.RATIONAL:
                # The full ns at scale 9; the chokepoint sets exactness per mode
                # (float rounds the double, rational stays exact).
                return cls._from_scaled_int(now_ns, 9, ctx.mode)
            case Mode.COMPLEX:
                raise NotRepresentableError("time() is a real clock reading; no complex value")
            case _:
                raise ValueError(f"unsupported mode: {ctx.mode!r}")

    @classmethod
    def vector(cls, elements: Sequence["Value"], element_mode: "Mode") -> "Value":
        """Build a one-dimensional VECTOR Value from already-evaluated elements (19.1.10).

        The elements are Values the node walk produced (the ``[a, b, …]`` literal's
        items), all in ``element_mode`` — the scalar mode the calculation runs in. An
        empty literal ``[]`` is allowed; ``element_mode`` is what tells it what it would
        hold. The vector is EXACT iff every element is exact (vacuously so when empty).

        Strictly one-dimensional: an element that is itself a vector is refused with a
        NotRepresentableError (an ArithmeticError, so the node walk attaches the line) —
        nesting like ``[[1,2],[3,4]]`` is not supported. The per-element mode match is
        enforced structurally by ``Vector``; this only adds the user-facing 1-D rule.
        """
        items = tuple(elements)
        if any(item.mode is Mode.VECTOR for item in items):
            raise NotRepresentableError(
                "vectors must be one-dimensional; a vector cannot hold a vector"
            )
        exact = all(item.exact for item in items)
        return cls(Mode.VECTOR, Vector(element_mode, items), exact=exact)

    # --- binary operators (19.3.1-19.3.7) -------------------------------
    # Each requires ``other`` in the SAME mode ("No mode mixing") and returns a
    # new Value ("Immutable"). Signatures may gain mode-specific arguments later.

    def _same_mode(self, other: "Value", op: str) -> bool:
        """Precondition shared by every binary op: a Value operand in this mode.

        Returns the propagated exactness (result exact only if BOTH operands
        were). Type dispatch stays in each operation's own match — this only
        enforces "operators take Values, nothing else" and "No mode mixing".
        """
        if not isinstance(other, Value):
            raise TypeError(f"Value {op} requires a Value, got {type(other).__name__}")
        if self.mode is Mode.VECTOR or other.mode is Mode.VECTOR:
            # Vectors are a container with no arithmetic yet (19.1.10): refuse every
            # binary op here, the one precondition all of them share, rather than
            # repeating an identical VECTOR arm in each operation's match. A clean
            # NotRepresentableError (ArithmeticError) so the node walk attaches the line.
            raise NotRepresentableError(f"vectors do not support {op}")
        if other.mode is not self.mode:
            raise TypeError(f"mode mismatch: {self.mode.value} {op} {other.mode.value}")
        return self.exact and other.exact

    def _fp_pair(self, other: "Value") -> tuple[FixedPoint, FixedPoint]:
        """Both operands' payloads as FixedPoints (the FIXED_POINT op helper).

        ``_same_mode`` has already proven both are this mode, so the payloads are
        FixedPoints; this just narrows the union for the type checker.
        """
        assert isinstance(self.payload, FixedPoint) and isinstance(other.payload, FixedPoint)
        return self.payload, other.payload

    def add(self, other: "Value") -> "Value":
        """Binary ``+`` (19.3.1)."""
        exact = self._same_mode(other, "+")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(Mode.FLOATING_POINT, self.payload + other.payload, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                num = a.mantissa * 10**b.decimals + b.mantissa * 10**a.decimals
                return _fp_value(num, 10 ** (a.decimals + b.decimals), a, b, exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                return Value(Mode.RATIONAL, self.payload + other.payload, exact=exact)
            case Mode.COMPLEX:
                return self._complex_add(other)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def sub(self, other: "Value") -> "Value":
        """Binary ``-`` (19.3.2)."""
        exact = self._same_mode(other, "-")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(Mode.FLOATING_POINT, self.payload - other.payload, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                num = a.mantissa * 10**b.decimals - b.mantissa * 10**a.decimals
                return _fp_value(num, 10 ** (a.decimals + b.decimals), a, b, exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                return Value(Mode.RATIONAL, self.payload - other.payload, exact=exact)
            case Mode.COMPLEX:
                return self._complex_sub(other)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def mul(self, other: "Value") -> "Value":
        """Binary ``*`` (19.3.3)."""
        exact = self._same_mode(other, "*")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(Mode.FLOATING_POINT, self.payload * other.payload, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                num = a.mantissa * b.mantissa
                return _fp_value(num, 10 ** (a.decimals + b.decimals), a, b, exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                return Value(Mode.RATIONAL, self.payload * other.payload, exact=exact)
            case Mode.COMPLEX:
                return self._complex_mul(other)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def div(self, other: "Value") -> "Value":
        """Binary ``/`` (19.3.4). Division by zero RAISES (provisional —
        follows Python, not IEEE ±inf; relates to 10.2.3)."""
        exact = self._same_mode(other, "/")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(Mode.FLOATING_POINT, self.payload / other.payload, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                if b.mantissa == 0:
                    raise ZeroDivisionError("fixed-point division by zero")
                # v1/v2 = (m1 * 10**d2) / (m2 * 10**d1); may not fit the scale.
                num, den = a.mantissa * 10**b.decimals, b.mantissa * 10**a.decimals
                return _fp_value(num, den, a, b, exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                return Value(Mode.RATIONAL, self.payload / other.payload, exact=exact)
            case Mode.COMPLEX:
                return self._complex_div(other)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def floordiv(self, other: "Value") -> "Value":
        """Binary ``//`` (19.3.5)."""
        exact = self._same_mode(other, "//")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(Mode.FLOATING_POINT, self.payload // other.payload, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                if b.mantissa == 0:
                    raise ZeroDivisionError("fixed-point floor division by zero")
                q = _fp_floor(a, b)  # floor(v1/v2), a whole number
                scale = max(a.decimals, b.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(q * 10**scale, scale), exact=exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                # Fraction // Fraction floors to an int; re-wrap as a Fraction.
                return Value(Mode.RATIONAL, Fraction(self.payload // other.payload), exact=exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("floor division is undefined for complex numbers")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def mod(self, other: "Value") -> "Value":
        """Binary ``%`` (19.3.6)."""
        exact = self._same_mode(other, "%")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(Mode.FLOATING_POINT, self.payload % other.payload, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                if b.mantissa == 0:
                    raise ZeroDivisionError("fixed-point modulo by zero")
                # v1 - floor(v1/v2)*v2, exact at the covering scale (no rounding).
                q = _fp_floor(a, b)
                scale = max(a.decimals, b.decimals)
                mant = a.mantissa * 10 ** (scale - a.decimals) - q * b.mantissa * 10 ** (
                    scale - b.decimals
                )
                return Value(Mode.FIXED_POINT, FixedPoint(mant, scale), exact=exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                return Value(Mode.RATIONAL, self.payload % other.payload, exact=exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("modulo is undefined for complex numbers")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def pow(self, other: "Value") -> "Value":
        """Binary ``**`` — POWER (19.3.7); also the ``pow(x, y)`` function (28.20).

        floating-point: ``x ** y`` (Python's float power); a negative base to a
            fractional exponent goes complex, which a double cannot hold, so it is
            a NotRepresentableError. Inexact when the operands are.
        rational: an INTEGER exponent only — exact (negative just inverts the
            Fraction); a non-integer exponent is irrational with no scale to round
            to, so it raises NotRepresentableError, the exact-or-refuse stance.
        fixed-point: an INTEGER exponent is EXACT via mantissa arithmetic (28.20)
            — negative inverts, zero base to a negative power is ZeroDivisionError.
            A FRACTIONAL exponent (28.20.1) is ``x**(p/q)``, the q-th root of
            ``x**p`` (a fixed-point exponent reduces to ``p/q`` with ``q`` a product
            of 2s and 5s), handled on a ladder:
              PATH A — exact-or-refuse: if the q-th root lands exactly on the grid
                (numerator AND denominator of ``x**p`` are perfect q-th powers), the
                result is that exact rational, quantized to the covering scale. This
                also reaches ODD roots of NEGATIVE bases ((-32)**0.2 == -2); an EVEN
                root of a negative base is complex -> NotRepresentableError.
              PATH B — inexact series: otherwise the root is irrational. For a
                POSITIVE base, ``x**y == exp(y*ln x)`` via the _fp_ln core (28.17)
                and an exp series, range-reduced by ln(2), quantized half-to-even
                (exact=False). A NEGATIVE base with an irrational root has no
                representable real value -> NotRepresentableError.
        """
        exact = self._same_mode(other, "**")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                # A negative base to a fractional exponent goes complex in
                # Python; floating-point cannot hold it, so it is a domain error.
                result = self.payload**other.payload
                if isinstance(result, complex):
                    raise NotRepresentableError("complex result in floating-point mode")
                return Value(Mode.FLOATING_POINT, result, exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                g = math.gcd(abs(b.mantissa), 10**b.decimals)
                p, q = b.mantissa // g, 10**b.decimals // g  # exponent as p/q, q > 0
                if q == 1:  # INTEGER exponent — the exact core (28.20)
                    if p >= 0:
                        num, den = a.mantissa**p, 10 ** (a.decimals * p)
                    elif a.mantissa == 0:
                        raise ZeroDivisionError("fixed-point zero to a negative power")
                    else:
                        num, den = 10 ** (a.decimals * -p), a.mantissa**-p
                    return _fp_value(num, den, a, b, exact)
                # FRACTIONAL exponent (28.20.1): x**(p/q) is the q-th root of x**p.
                scale = max(a.decimals, b.decimals)  # the covering result scale
                if a.mantissa == 0:  # 0**(p/q): 0 for p > 0, undefined for p < 0
                    if p < 0:
                        raise ZeroDivisionError("fixed-point zero to a negative power")
                    return Value(Mode.FIXED_POINT, FixedPoint(0, scale), exact=exact)
                # PATH A — an EXACT perfect q-th root stays on the grid.
                xp = Fraction(a.mantissa, 10**a.decimals) ** p  # x**p, exact rational
                num, den = xp.numerator, xp.denominator  # den > 0, lowest terms
                sign = 1
                if num < 0:  # only when base < 0 and p is odd
                    if q % 2 == 0:  # an even root of a negative is complex
                        raise NotRepresentableError("even root of a negative value")
                    sign, num = -1, -num
                root_num, root_den = _perfect_root(num, q), _perfect_root(den, q)
                if root_num is not None and root_den is not None:
                    return _fp_value(sign * root_num, root_den, a, b, exact)
                # PATH B — irrational root via exp(y*ln x); needs a positive base.
                if a.mantissa < 0:
                    raise NotRepresentableError("fractional power of a negative base is irrational")
                ln_total, working = _fp_ln(a)  # ln(x) * 10**working
                unity = 10**working
                w = ln_total * p // q  # y*ln(x) at the working scale
                ln2 = _ln2_scaled(working)
                k = w // ln2  # exp(w) == 2**k * exp(s), s in [0, ln 2)
                s = w - k * ln2
                es = _fp_exp_series(s, unity)  # exp(s) * unity, s >= 0 and small
                num, den = (es << k, unity) if k >= 0 else (es, unity << -k)
                mantissa, _ = _fp_quantize(num, den, scale)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=False)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                # A rational raised to a non-integer exponent is irrational —
                # not representable as a Fraction, so it is a domain error.
                if other.payload.denominator != 1:
                    raise NotRepresentableError("rational power requires an integer exponent")
                return Value(Mode.RATIONAL, self.payload ** int(other.payload), exact=exact)
            case Mode.COMPLEX:
                return self._complex_pow(other)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    # --- bitwise operators (24.3.2) -------------------------------------
    # Available in EVERY mode, acting on that mode's own stored bits: the 64-bit
    # IEEE-754 pattern for floating-point, the (scale-aligned) mantissa for
    # fixed-point, and the numerator/denominator pair for rational. The three
    # binary ops differ only in the underlying int operator, so they share one
    # per-mode match in _bitwise; the named methods stay (no operator dunders, 19.5).

    def _bitwise(self, other: "Value", symbol: str, intop: Callable[[int, int], int]) -> "Value":
        """Shared body of ``&`` / ``|`` / ``^`` — apply ``intop`` to each mode's bits.

        Bit manipulation is exact (no rounding), so the result's exactness is just
        the propagated operand exactness. Rational combines numerator and
        denominator independently (the raw storage pair); ``intop`` on the
        denominators may land on 0 (e.g. ``2 & 1``), which Fraction rejects as a
        ZeroDivisionError — a domain error BinOp.evaluate tags with the line (18.4).
        """
        exact = self._same_mode(other, symbol)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                bits = intop(_float_bits(self.payload), _float_bits(other.payload))
                return Value(Mode.FLOATING_POINT, _bits_float(bits), exact=exact)
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)
                ma, mb, scale = _fp_align(a, b)
                return Value(Mode.FIXED_POINT, FixedPoint(intop(ma, mb), scale), exact=exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                p, q = self.payload, other.payload
                num, den = intop(p.numerator, q.numerator), intop(p.denominator, q.denominator)
                return Value(Mode.RATIONAL, Fraction(num, den), exact=exact)
            case Mode.COMPLEX:
                raise NotRepresentableError(f"bitwise {symbol} is undefined for complex numbers")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def bitand(self, other: "Value") -> "Value":
        """Binary ``&`` — bitwise AND (24.3.2)."""
        return self._bitwise(other, "&", operator.and_)

    def bitor(self, other: "Value") -> "Value":
        """Binary ``|`` — bitwise OR (24.3.2)."""
        return self._bitwise(other, "|", operator.or_)

    def bitxor(self, other: "Value") -> "Value":
        """Binary ``^`` — bitwise XOR, NOT power (24.3.2; power is ``**``)."""
        return self._bitwise(other, "^", operator.xor)

    # --- unary operators (19.3.8-19.3.9) --------------------------------

    def neg(self) -> "Value":
        """Unary ``-`` (19.3.8)."""
        match self.mode:
            case Mode.FLOATING_POINT | Mode.RATIONAL:
                assert isinstance(self.payload, (float, Fraction))
                return Value(self.mode, -self.payload, exact=self.exact)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                return Value(self.mode, FixedPoint(-fp.mantissa, fp.decimals), exact=self.exact)
            case Mode.COMPLEX:
                return self._complex_neg()
            case Mode.VECTOR:
                raise NotRepresentableError("vectors do not support unary -")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def pos(self) -> "Value":
        """Unary ``+`` (19.3.9)."""
        match self.mode:
            case Mode.FLOATING_POINT | Mode.RATIONAL:
                assert isinstance(self.payload, (float, Fraction))
                return Value(self.mode, +self.payload, exact=self.exact)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                return Value(self.mode, FixedPoint(+fp.mantissa, fp.decimals), exact=self.exact)
            case Mode.COMPLEX:
                return self._complex_pos()
            case Mode.VECTOR:
                raise NotRepresentableError("vectors do not support unary +")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def bitnot(self) -> "Value":
        """Unary ``~`` — bitwise NOT (24.3.2), on the mode's own bits.

        Python-native ``~x == -x-1``: floating-point flips all 64 IEEE-754 bits
        (the mask in _bits_float folds the sign back into the pattern); fixed-point
        inverts the mantissa at its own scale; rational inverts numerator and
        denominator independently (``~den`` is <= -2, never 0, so no division
        trap). Exact bit manipulation, so exactness carries through unchanged.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                flipped = _bits_float(~_float_bits(self.payload))
                return Value(Mode.FLOATING_POINT, flipped, exact=self.exact)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                flipped_fp = FixedPoint(~fp.mantissa, fp.decimals)
                return Value(Mode.FIXED_POINT, flipped_fp, exact=self.exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                fr = self.payload
                inverted = Fraction(~fr.numerator, ~fr.denominator)
                return Value(Mode.RATIONAL, inverted, exact=self.exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("bitwise NOT is undefined for complex numbers")
            case Mode.VECTOR:
                raise NotRepresentableError("vectors do not support bitwise ~")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    # --- function methods (19.5) ----------------------------------------
    # The per-mode home for section-22's call functions, each a named method
    # with its own match (like the operators), dispatched from nodes._FUNCS.
    # Per-mode semantics live in the method's docstring (the spec).

    def abs_(self) -> "Value":
        """Absolute value (19.5.1) — exact in every mode, the shape of neg().

        Magnitude only ever drops a sign, never crosses the mode's grid, so it is
        representable everywhere and exactness carries through unchanged. The
        trailing underscore keeps the name off the ``abs`` builtin used below.
        """
        match self.mode:
            case Mode.FLOATING_POINT | Mode.RATIONAL:
                assert isinstance(self.payload, (float, Fraction))
                # .__abs__() not abs(): the builtin joins the float|Fraction union to
                # object, the dunder distributes over it and keeps the type (and -0.0).
                return Value(self.mode, self.payload.__abs__(), exact=self.exact)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                return Value(self.mode, FixedPoint(abs(fp.mantissa), fp.decimals), exact=self.exact)
            case Mode.COMPLEX:
                return self._complex_abs()
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def conj(self) -> "Value":
        """Complex conjugate a-bi (40.12). A real value is its own conjugate, so in
        every non-complex mode this is the identity (returned verbatim, exactness
        intact); in complex mode it negates the imaginary part."""
        if self.mode is Mode.COMPLEX:
            return self._complex_conj()
        return self

    def re(self) -> "Value":
        """Real part Re(z) (40.12). The identity in a real mode (the value IS its real
        part); in complex mode the real part as a complex value with zero imaginary."""
        if self.mode is Mode.COMPLEX:
            return self._complex_re()
        return self

    def im(self) -> "Value":
        """Imaginary part Im(z) (40.12). Zero in every real mode (a real has no
        imaginary part); in complex mode the imaginary magnitude as a complex value."""
        if self.mode is Mode.COMPLEX:
            return self._complex_im()
        return Value._from_scaled_int(0, 0, self.mode)

    def arg(self) -> "Value":
        """Argument/phase arg(z) in radians (40.12) — atan2(Im z, Re z). In a real
        mode this is atan2(0, x): 0 for x >= 0, pi for x < 0, inheriting atan2's
        per-mode exactness (rational refuses the irrational pi). Complex mode takes
        the angle of the two parts."""
        if self.mode is Mode.COMPLEX:
            return self._complex_arg()
        return Value._from_scaled_int(0, 0, self.mode).atan2(self)

    def sign(self) -> "Value":
        """Signum (40.9) — UNARY classification: -1, 0, or +1 by the operand's sign,
        in the mode's own spelling of those three integers.

        A COMPARISON to zero, not arithmetic, so unlike gcd/lcm it works on ANY value
        — integer or not — and never leaves the grid. The DECISION (40.9) is that the
        result is EXACT regardless of the operand's flag: it does NOT carry the
        inexact flag the way ``abs`` does, because ``abs`` transforms the value (so an
        inexact input stays inexact) whereas ``sign`` reports the sign of the STORED
        value, which is certain — one of three exactly-representable integers. So an
        inexact operand still yields an exact sign in fixed-point and rational.
        Floating-point is the lone exception: like every binary64 result here (cf.
        ``gcd``) it carries the unconditional inexact flag, even though -1.0/0.0/+1.0
        are representable. ``sign(0)`` is 0 (and float ``-0.0`` classifies as 0).
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                s = (self.payload > 0.0) - (self.payload < 0.0)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                m = self.payload.mantissa
                s = (m > 0) - (m < 0)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                s = (self.payload > 0) - (self.payload < 0)
            case Mode.COMPLEX:
                raise NotRepresentableError("sign is undefined for complex numbers (no ordering)")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")
        return Value._from_scaled_int(s, 0, self.mode)

    @staticmethod
    def _as_ndigits(value: "Value") -> int:
        """Read the optional ``ndigits`` argument of the rounding family as a Python int.

        The second operand of floor/ceil/round/trunc (28.22-28.26) is a COUNT of
        decimal places, not a value-in-the-mode: it is evaluated like any operand,
        then required to be an INTEGER (no fractional part) in whatever mode the run
        is in and read as a plain ``int``. A non-integer refuses — the same
        integer-argument stance ``pow`` takes for its exponent (28.20). The exactness
        FLAG is irrelevant (a float ``2.0`` is inexact here yet a valid count); only
        the value's integrality matters. A negative count is allowed (it rounds to
        tens/hundreds), so no sign check.
        """
        match value.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(value.payload, float)
                if not value.payload.is_integer():
                    raise NotRepresentableError("ndigits must be an integer")
                return int(value.payload)
            case Mode.FIXED_POINT:
                assert isinstance(value.payload, FixedPoint)
                fp = value.payload
                whole, frac = divmod(fp.mantissa, 10**fp.decimals)
                if frac != 0:
                    raise NotRepresentableError("ndigits must be an integer")
                return whole
            case Mode.RATIONAL:
                assert isinstance(value.payload, Fraction)
                if value.payload.denominator != 1:
                    raise NotRepresentableError("ndigits must be an integer")
                return value.payload.numerator
            case _:
                raise ValueError(f"unsupported mode: {value.mode!r}")

    def floor(self, ndigits: "Value | None" = None) -> "Value":
        """Round toward NEGATIVE infinity (28.23) — the shape of ``abs`` (19.5.1)
        with an optional ``ndigits`` count (28.22): a required operand plus one
        optional trailing integer that fixes how many decimal places to floor at
        (default 0 — floor to a whole number). ``floor(2.7) -> 2``,
        ``floor(-2.1) -> -3``; ``floor(1234, -2) -> 1200`` (negative ndigits rounds
        to tens/hundreds, Python's semantics).

        EXACT in every mode — flooring SELECTS a representable value, it does not
        compute — EXCEPT floating-point with ``ndigits > 0``, where the n-decimal
        target (e.g. 2.70) is not binary-representable; flooring a float to an
        INTEGER (ndigits <= 0) lands on a whole multiple a double holds exactly, so
        it stays exact.
        floating-point: ``math.floor`` to an int (ndigits <= 0, exact), else the
            decimal shift ``floor(x*10**n)/10**n`` (inexact).
        fixed-point: ``M // 10**d`` on the scaled mantissa — Python ``//`` already
            floors toward -inf — held at scale ``max(0, n)``. Exact.
        rational: ``math.floor`` of the shifted fraction, back to a Fraction. Exact.
        """
        n = 0 if ndigits is None else Value._as_ndigits(ndigits)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if n <= 0:
                    # Floor to a whole multiple of 10**-n: an integer-valued double,
                    # exactly representable. Pure-int scaling keeps it exact.
                    unit = 10**-n
                    whole = math.floor(self.payload / unit) * unit
                    return Value(Mode.FLOATING_POINT, float(whole), exact=True)
                # n > 0: the n-decimal target is not binary-representable -> inexact.
                unit = 10**n
                shifted = math.floor(self.payload * unit) / unit
                return Value(Mode.FLOATING_POINT, shifted, exact=False)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                # Mantissa floored to scale n (Python // floors toward -inf), then
                # rescaled to the result scale max(0, n): for n < 0 the floor lands
                # on tens/hundreds carried at scale 0.
                if n >= fp.decimals:  # finer than the value: nothing to drop
                    at_n = fp.mantissa * 10 ** (n - fp.decimals)
                else:
                    at_n = fp.mantissa // 10 ** (fp.decimals - n)
                scale = max(0, n)
                mantissa = at_n * 10 ** (scale - n)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=self.exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                shift = Fraction(10) ** n
                floored = Fraction(math.floor(self.payload * shift)) / shift
                return Value(Mode.RATIONAL, floored, exact=self.exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("floor is undefined for complex numbers (no ordering)")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def ceil(self, ndigits: "Value | None" = None) -> "Value":
        """Round toward POSITIVE infinity (28.24) — the mirror of ``floor`` (28.23),
        same shape (operand + optional ``ndigits`` count, default 0) and same
        per-mode exactness story. ``ceil(2.1) -> 3``, ``ceil(-2.7) -> -2``;
        ``ceil(1234, -2) -> 1300`` (negative ndigits rounds to tens/hundreds).

        EXACT in every mode — ceiling SELECTS a representable value, no compute —
        EXCEPT floating-point with ``ndigits > 0``, where the n-decimal target is
        not binary-representable; ceiling a float to an INTEGER (ndigits <= 0) lands
        on a whole multiple a double holds exactly, so it stays exact.
        floating-point: ``math.ceil`` to an int (ndigits <= 0, exact), else the
            decimal shift ``ceil(x*10**n)/10**n`` (inexact).
        fixed-point: ``-((-M) // 10**d)`` on the scaled mantissa — ceiling is
            ``-floor(-x)`` — held at scale ``max(0, n)``. Exact.
        rational: ``math.ceil`` of the shifted fraction, back to a Fraction. Exact.
        """
        n = 0 if ndigits is None else Value._as_ndigits(ndigits)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if n <= 0:
                    # Ceil to a whole multiple of 10**-n: an integer-valued double,
                    # exactly representable. Pure-int scaling keeps it exact.
                    unit = 10**-n
                    whole = math.ceil(self.payload / unit) * unit
                    return Value(Mode.FLOATING_POINT, float(whole), exact=True)
                # n > 0: the n-decimal target is not binary-representable -> inexact.
                unit = 10**n
                shifted = math.ceil(self.payload * unit) / unit
                return Value(Mode.FLOATING_POINT, shifted, exact=False)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                # Mantissa ceiled to scale n via -((-M) // 10**d) (-floor(-x)), then
                # rescaled to the result scale max(0, n): for n < 0 the ceil lands on
                # tens/hundreds carried at scale 0.
                if n >= fp.decimals:  # finer than the value: nothing to drop
                    at_n = fp.mantissa * 10 ** (n - fp.decimals)
                else:
                    at_n = -((-fp.mantissa) // 10 ** (fp.decimals - n))
                scale = max(0, n)
                mantissa = at_n * 10 ** (scale - n)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=self.exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                shift = Fraction(10) ** n
                ceiled = Fraction(math.ceil(self.payload * shift)) / shift
                return Value(Mode.RATIONAL, ceiled, exact=self.exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("ceil is undefined for complex numbers (no ordering)")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def round_(self, ndigits: "Value | None" = None) -> "Value":
        """Round to NEAREST, ties to EVEN (28.25) — same operand + optional
        ``ndigits`` shape as ``floor``/``ceil`` (28.23/28.24), but rounding to the
        closest representable value, a tie going to the even neighbour (banker's
        rounding): ``round(2.5) -> 2``, ``round(3.5) -> 4``, ``round(-2.5) -> -2``;
        ``round(2.345, 2) -> 2.34``. Half-to-even matches Python's builtin ``round``,
        the IEEE-754 default, and the engine's own fixed-point quantiser (sqrt,
        22.4.2) — the whole engine rounds ONE way. The trailing underscore keeps the
        name off the ``round`` builtin used below, as ``abs_`` does for ``abs``.

        EXACT in every mode — rounding SELECTS a representable value, no compute —
        EXCEPT floating-point with ``ndigits > 0``, where the n-decimal target is
        not binary-representable; rounding a float to an INTEGER (ndigits <= 0) lands
        on a whole multiple a double holds exactly, so it stays exact.
        floating-point: builtin ``round(x, n)`` (already half-even); exact for
            ndigits <= 0, inexact otherwise.
        fixed-point: the engine's half-even quantiser on the scaled mantissa at the
            target scale, held at scale ``max(0, n)``. Exact.
        rational: ``round(Fraction, n)`` — Fraction.__round__ is half-even — back to
            a Fraction. Exact.
        """
        n = 0 if ndigits is None else Value._as_ndigits(ndigits)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                # round(x, n) is half-even and returns a float for any int n; n <= 0
                # lands on an integer-valued double (exact), n > 0 on a non-binary
                # n-decimal target (inexact).
                return Value(Mode.FLOATING_POINT, round(self.payload, n), exact=n <= 0)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                # Half-even round the mantissa to scale n via the engine quantiser
                # (the same _fp_quantize sqrt/div use), then rescale to the result
                # scale max(0, n): for n < 0 the round lands on tens/hundreds at scale 0.
                if n >= fp.decimals:  # finer than the value: nothing to drop
                    at_n = fp.mantissa * 10 ** (n - fp.decimals)
                else:
                    at_n, _ = _fp_quantize(fp.mantissa, 10 ** (fp.decimals - n), 0)
                scale = max(0, n)
                mantissa = at_n * 10 ** (scale - n)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=self.exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                return Value(Mode.RATIONAL, Fraction(round(self.payload, n)), exact=self.exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("round is undefined for complex numbers (no ordering)")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def trunc(self, ndigits: "Value | None" = None) -> "Value":
        """Round toward ZERO (28.26) — drop the fraction: ``floor`` for x >= 0,
        ``ceil`` for x < 0, the same operand + optional ``ndigits`` shape and
        exactness story as the rest of the family (28.23-28.25). ``trunc(2.7) -> 2``,
        ``trunc(-2.7) -> -2`` (vs ``floor(-2.7) -> -3``); ``trunc(1290, -2) -> 1200``.
        This is the explicit toward-zero function over ANY value — the same idea as
        fixed-point's literal mantissa truncation in ``time`` (28.1.1), generalised.

        EXACT in every mode — truncation SELECTS a representable value, no compute —
        EXCEPT floating-point with ``ndigits > 0``, where the n-decimal target is
        not binary-representable; truncating a float to an INTEGER (ndigits <= 0)
        lands on a whole multiple a double holds exactly, so it stays exact.
        floating-point: ``math.trunc`` to an int (ndigits <= 0, exact), else the
            decimal shift ``trunc(x*10**n)/10**n`` (inexact).
        fixed-point: ``sign * (abs(M) // 10**d)`` on the scaled mantissa — magnitude
            floored, sign restored — held at scale ``max(0, n)``. Exact.
        rational: ``math.trunc`` of the shifted fraction, back to a Fraction. Exact.
        """
        n = 0 if ndigits is None else Value._as_ndigits(ndigits)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if n <= 0:
                    # Truncate to a whole multiple of 10**-n: an integer-valued
                    # double, exactly representable. Pure-int scaling keeps it exact.
                    unit = 10**-n
                    whole = math.trunc(self.payload / unit) * unit
                    return Value(Mode.FLOATING_POINT, float(whole), exact=True)
                # n > 0: the n-decimal target is not binary-representable -> inexact.
                unit = 10**n
                shifted = math.trunc(self.payload * unit) / unit
                return Value(Mode.FLOATING_POINT, shifted, exact=False)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                # Mantissa truncated to scale n via sign * (abs(M) // 10**d) — floor
                # the magnitude, restore the sign (toward zero) — then rescale to the
                # result scale max(0, n): for n < 0 it lands on tens/hundreds at scale 0.
                if n >= fp.decimals:  # finer than the value: nothing to drop
                    at_n = fp.mantissa * 10 ** (n - fp.decimals)
                else:
                    sign = -1 if fp.mantissa < 0 else 1
                    at_n = sign * (abs(fp.mantissa) // 10 ** (fp.decimals - n))
                scale = max(0, n)
                mantissa = at_n * 10 ** (scale - n)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=self.exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                shift = Fraction(10) ** n
                truncated = Fraction(math.trunc(self.payload * shift)) / shift
                return Value(Mode.RATIONAL, truncated, exact=self.exact)
            case Mode.COMPLEX:
                raise NotRepresentableError("trunc is undefined for complex numbers (no ordering)")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def sum_(self, *others: "Value") -> "Value":
        """Total of one-or-more operands — VARIADIC; repeated ``+``.

        ``self`` is the first operand, so ``sum_(a)`` is just ``a`` (the empty fold)
        and ``sum_(a, b, c)`` is ``((a + b) + c)``. Folding over ``add`` (19.3.1)
        rather than re-deriving per-mode math means every binary-op contract comes
        for free: same-mode enforcement (no mixing), the covering fixed-point scale,
        and exactness propagation. So like repeated ``+`` it is EXACT in every mode
        — addition never leaves the grid — except where an operand was already
        inexact, which the fold carries through. Internal now: ``avg`` (28.4) builds
        its total here. The ``sum`` BUILTIN is no longer this variadic operand-method
        — it became the range-fold special form ``sum(i, lo, hi, expr)`` (40.19,
        expr.forms) — but the helper stays (avg needs it); the trailing underscore
        keeps the name off Python's ``sum``, as ``abs_`` does for ``abs``.
        """
        return functools.reduce(Value.add, others, self)

    def avg(self, *others: "Value") -> "Value":
        """Arithmetic mean (28.4) — VARIADIC; ``sum / count``.

        Builds the total with ``sum_`` (exact like repeated ``+``), then divides by
        the operand count carried as a same-mode whole number, so the result follows
        the mode's own ``/`` rule (19.3.4): fixed-point quantizes to the covering
        scale and may be inexact, rational is exact, float rounds. ``avg(a)`` is
        ``a / 1`` — the single-operand identity (modulo float's inexact flag).
        """
        total = self.sum_(*others)
        count = Value._from_scaled_int(1 + len(others), 0, self.mode)
        return total.div(count)

    def _compare(self, other: "Value", op: str) -> int:
        """Order two same-mode Values: -1 if self < other, 0 if equal, +1 if greater.

        The ordering primitive behind max/min (28.2/28.3). It is VALUE-only, so a
        fixed-point pair compares across scales (1.5 == 1.50) by cross-multiplying
        — the same widening ``add``/``//`` use. Same-mode is enforced like every
        binary op; the exactness ``_same_mode`` propagates is irrelevant to an
        order and discarded.
        """
        self._same_mode(other, op)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return (self.payload > other.payload) - (self.payload < other.payload)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                return (self.payload > other.payload) - (self.payload < other.payload)
            case Mode.FIXED_POINT:
                x, y = self._fp_pair(other)
                a = x.mantissa * 10**y.decimals
                b = y.mantissa * 10**x.decimals
                return (a > b) - (a < b)
            case Mode.COMPLEX:
                # No total order on C — so min/max/median/clamp, which all route
                # through here, refuse rather than invent a magnitude/real-part order.
                raise NotRepresentableError(f"{op} is undefined for complex numbers (no ordering)")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def _selection_operands(self, others: tuple["Value", ...], op: str) -> tuple["Value", ...]:
        """The operands a selection aggregate (max/min, 28.2/28.3) ranges over.

        Two call shapes. The usual one is a FLAT run of scalars — ``self`` plus
        ``others`` — returned unchanged. The other is a SINGLE vector (19.1.10):
        ``max([a, b, …])`` ranges over the vector's ELEMENTS, reducing a list rather
        than its arguments. Either way the caller then selects one operand verbatim.

        These are two OVERLOADS, not a blend: a vector is legal only as the SOLE
        operand, so mixing it with anything else (``max([1, 2], 3)`` or two vectors)
        refuses with a message that spells the two forms out — it defers the
        multi-vector call shape (40.13). An EMPTY vector has nothing to select from,
        so it refuses too — a maximum/minimum of no values is undefined.
        """
        operands = (self, *others)
        if not any(v.mode is Mode.VECTOR for v in operands):
            return operands
        if len(operands) > 1:
            raise NotRepresentableError(
                f"{op} has two forms — {op}(vector) or {op}(a, b, …) — and cannot mix them"
            )
        assert isinstance(self.payload, Vector)
        if not self.payload.elements:
            raise NotRepresentableError(f"{op} of an empty vector is undefined")
        return self.payload.elements

    def max_(self, *others: "Value") -> "Value":
        """Largest of one-or-more operands (28.2) — VARIADIC; SELECTION, not math.

        Like ``sum_``, ``self`` is the first operand, so ``max(a)`` is ``a``. A
        single vector operand is reduced over its elements instead — ``max([a, b,
        …])`` is the largest element (19.1.10); see ``_selection_operands`` for the
        vector call shape. It never computes: it returns whichever operand is
        greatest VERBATIM, so the result carries that operand's own scale and
        exactness — no covering, no rounding, exact iff the chosen operand was.
        Comparison is value-only (``_compare``, same-mode-enforced); a tie keeps the
        EARLIER operand (strict ``>``), so the choice is stable. Mirror: ``min_`` (28.3).
        """
        operands = self._selection_operands(others, "max")
        best = operands[0]
        for other in operands[1:]:
            if other._compare(best, "max") > 0:
                best = other
        return best

    def min_(self, *others: "Value") -> "Value":
        """Smallest of one-or-more operands (28.3) — the mirror of ``max_`` (28.2):
        variadic, selection-only (so exact, carrying the chosen operand's scale and
        exactness), same-mode, and ties keep the earlier operand (strict ``<``). A
        single vector operand is reduced over its elements — ``min([a, b, …])`` is
        the smallest element (19.1.10), the same vector call shape as ``max_``.
        """
        operands = self._selection_operands(others, "min")
        best = operands[0]
        for other in operands[1:]:
            if other._compare(best, "min") < 0:
                best = other
        return best

    def median(self, *others: "Value") -> "Value":
        """Middle operand by value (28.7) — VARIADIC; order-only, then maybe average.

        Sorts the operands by ``_compare`` (value-only, same-mode-enforced — no
        arithmetic, so the ordering never rounds). An ODD count has a single middle,
        returned VERBATIM like ``max_``/``min_`` (28.2/28.3): pure selection, so it
        carries the chosen operand's own scale and exactness. An EVEN count averages
        the two straddling middles — ``(lo + hi) / 2`` — which COMPUTES, so it
        follows the mode's ``/`` rule exactly as ``avg`` (28.4): fixed-point may
        round, rational is exact, float rounds. ``median(a)`` is ``a`` (odd count 1).
        """

        def by_value(a: "Value", b: "Value") -> int:
            return a._compare(b, "median")

        ordered = sorted((self, *others), key=functools.cmp_to_key(by_value))
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        two = Value._from_scaled_int(2, 0, self.mode)
        return ordered[mid - 1].add(ordered[mid]).div(two)

    def clamp(self, lo: "Value", hi: "Value") -> "Value":
        """Constrain x to the range [lo, hi] = min(hi, max(lo, x)) (40.21) — TERNARY
        fixed-arity-3; ``self`` is x, ``lo``/``hi`` the bounds.

        SELECTION, not math, like max/min (28.2/28.3): it never computes, returning
        whichever of the three operands applies VERBATIM — so the result carries that
        operand's own scale and exactness, never rounds, and is EXACT in EVERY mode
        (the sign stance, 40.9). Comparison is value-only (``_compare``,
        same-mode-enforced across all three), so a fixed-point bound compares across
        scales. DOMAIN ``lo <= hi``: an inverted range is meaningless and REFUSES. x
        below lo gives lo, above hi gives hi, otherwise x is returned untouched (a
        boundary tie returns the bound's value via x, x being within range).
        """
        if lo._compare(hi, "clamp") > 0:
            raise NotRepresentableError("clamp requires lo <= hi")
        if self._compare(lo, "clamp") < 0:  # x < lo
            return lo
        if self._compare(hi, "clamp") > 0:  # x > hi
            return hi
        return self

    def lerp(self, b: "Value", t: "Value") -> "Value":
        """Linear interpolation a + (b - a)*t (40.22) — TERNARY fixed-arity-3;
        ``self`` is a (the t=0 endpoint), ``b`` the t=1 endpoint, ``t`` the fraction.

        Plain ARITHMETIC, not selection: composed straight from sub/mul/add, so it
        inherits their per-mode stance exactly (the avg/division family, 28.4) —
        EXACT in rational, and MAY ROUND in fixed-point/float where the ``*t``
        multiply leaves the grid; same-mode is enforced by the underlying ops. ``t``
        is unrestricted: t in [0, 1] interpolates between the endpoints, t outside
        extrapolates past them (t=0 gives a, t=1 gives b).
        """
        return self.add(b.sub(self).mul(t))

    def variance(self, *others: "Value") -> "Value":
        """POPULATION variance (28.8) — VARIADIC; sum of squared deviations / n.

        DIVIDES BY n (the population convention, not the sample n-1; a sample
        variant is still to plan). Composes the aggregates already built: the mean
        is ``avg`` (28.4), each deviation ``op - mean`` is squared with ``mul``, and
        their total (repeated ``+``) is divided by the count. Both the mean and the
        final divide COMPUTE, so the result follows the mode's ``/`` rule twice over:
        fixed-point may round (and a rounded mean compounds the loss — work at a
        wider scale via min_fixed_point_precision for accuracy), rational is exact,
        float rounds. ``variance(a)`` is ``0`` — a lone point has no spread.
        """
        operands = (self, *others)
        mean = self.avg(*others)
        deviations = (op.sub(mean) for op in operands)
        total = functools.reduce(Value.add, (d.mul(d) for d in deviations))
        count = Value._from_scaled_int(len(operands), 0, self.mode)
        return total.div(count)

    def stddev(self, *others: "Value") -> "Value":
        """POPULATION standard deviation (28.9) — VARIADIC; ``sqrt(variance)``.

        The square root of the population ``variance`` (28.8), so it INHERITS
        ``sqrt``'s per-mode story (19.5.2): unconditionally inexact in float and
        fixed-point, and in RATIONAL mode it raises NotRepresentableError when the
        root is irrational rather than fabricate digits — exact there only when the
        variance is a perfect-square fraction. ``stddev(a)`` is ``sqrt(0) = 0``.
        """
        return self.variance(*others).sqrt()

    def pct(self, p: "Value") -> "Value":
        """``p`` percent OF ``self`` (36.1) — ``self * p / 100``.

        The everyday percentage op, named so the caller never hand-rolls the
        ``/ 100`` (the off-by-100 / 25-bps-read-as-25% class). It COMPUTES, so it
        composes the engine's own ``*`` and ``/`` (19.3.3/19.3.4): the multiply
        stays on the grid, then the divide by a same-mode whole ``100`` follows the
        active mode's ``/`` rule — fixed-point quantizes to the covering scale and
        may be inexact, rational is exact, float rounds. Same-mode is enforced by
        the composed ``*``. ``pct(x, 100)`` is ``x`` and ``pct(x, 0)`` is ``0``.
        For basis points use ``bps`` (36.2).
        """
        hundred = Value._from_scaled_int(100, 0, self.mode)
        return self.mul(p).div(hundred)

    def pct_change(self, new: "Value") -> "Value":
        """Signed relative change from ``self`` (old) to ``new`` (36.1) —
        ``(new - old) / old``, a fraction (multiply by 100, or feed to ``pct``, for
        a percentage).

        It COMPUTES, composing ``-`` then ``/`` (19.3.2/19.3.4), so it follows the
        active mode's ``/`` rule: fixed-point quantizes and may be inexact, rational
        is exact, float rounds. Same-mode is enforced by the composed ops. The old
        value (``self``) is the denominator, so ``pct_change(0, x)`` divides by zero
        and RAISES like any ``/`` by zero; ``pct_change(x, x)`` is ``0``.
        """
        return new.sub(self).div(self)

    def bps(self, b: "Value") -> "Value":
        """``b`` basis points OF ``self`` (36.2) — ``self * b / 10000``.

        The basis-point twin of ``pct`` (36.1), existing SOLELY to make the
        bps-vs-percent distinction un-confusable at the call site (the 25-bps read
        as 25% class — 25 bps is 0.25%). It COMPUTES, composing the engine's own
        ``*`` and ``/`` (19.3.3/19.3.4): the multiply stays on the grid, then the
        divide by a same-mode whole ``10000`` follows the active mode's ``/`` rule —
        fixed-point quantizes to the covering scale and may be inexact, rational is
        exact, float rounds. Same-mode is enforced by the composed ``*``.
        ``bps(x, 10000)`` is ``x`` and ``bps(x, 100)`` equals ``pct(x, 1)``.
        """
        myriad = Value._from_scaled_int(10000, 0, self.mode)
        return self.mul(b).div(myriad)

    def compound(self, rate: "Value", periods: "Value") -> "Value":
        """Compound growth of ``self`` (the principal) at a PER-PERIOD ``rate`` over
        ``periods`` periods (36.3) — ``principal * (1 + rate)**periods``.

        Named so the period is EXPLICIT: ``rate`` is the growth PER PERIOD and
        ``periods`` counts the SAME unit, so an annual rate can never silently act
        monthly (the rate-period confusion class — quote a monthly rate with a month
        count). It COMPUTES, composing ``+`` to build the per-period factor
        ``1 + rate``, ``**`` (19.3.7) to raise it, then ``*`` to scale the principal,
        so it inherits each mode's rule: with a WHOLE ``periods`` the power is exact
        via mantissa arithmetic (fixed-point) or an integer Fraction power
        (rational), and only the final multiply may quantize; a FRACTIONAL
        ``periods`` rides ``**``'s fractional-exponent ladder (28.20.1) — exact on a
        perfect root, otherwise an inexact series (rational REFUSES an irrational
        power, the exact-or-refuse stance). Same-mode is enforced by the composed
        ops. ``compound(p, r, 0)`` is ``p`` (any base to the 0th power).
        """
        one = Value._from_scaled_int(1, 0, self.mode)
        factor = one.add(rate).pow(periods)
        return self.mul(factor)

    def _is_zero(self) -> bool:
        """True when this Value equals zero (value-only, scale-blind), the rate==0
        guard for the time-value-of-money trio (36.4) whose closed forms otherwise
        divide by the rate. Compares with ``_compare`` so ``0.00`` and ``0`` agree.
        """
        zero = Value._from_scaled_int(0, 0, self.mode)
        return self._compare(zero, "tvm") == 0

    def pmt(self, nper: "Value", pv: "Value") -> "Value":
        """Level PAYMENT amortising a present value ``pv`` to zero over ``nper``
        periods at the PER-PERIOD rate ``self`` (36.4) — the loan/amortisation case.

        The time-value-of-money trio (``pmt``/``fv``/``pv``) is the ordinary-annuity
        convention: a payment at each period END, a residual value of 0, and NO sign
        flip — a positive loan gives a positive payment (unlike a spreadsheet). The
        rate is PER PERIOD and ``nper`` counts the SAME unit, the rate-period story
        ``compound`` (36.3) tells. Closed form
        ``pmt = pv * r / (1 - (1 + r)**-nper)``; with a ZERO rate there is no
        interest, so it is the principal split evenly, ``pv / nper``. It COMPUTES,
        composing the engine's own +, **, -, * and / so it inherits each mode's rule:
        the negative-integer power inverts exactly (fixed-point/rational), the
        divides follow the mode's ``/`` (fixed-point quantizes and may be inexact,
        rational is exact, float rounds). ``nper`` of 0 has no periods to pay over and
        divides by zero like any other (the amortisation is undefined).
        """
        if self._is_zero():
            return pv.div(nper)
        one = Value._from_scaled_int(1, 0, self.mode)
        denom = one.sub(one.add(self).pow(nper.neg()))
        return pv.mul(self).div(denom)

    def fv(self, nper: "Value", pmt: "Value") -> "Value":
        """FUTURE value of an ``nper``-period stream of level payments ``pmt`` at the
        PER-PERIOD rate ``self`` (36.4) — the ordinary-annuity accumulation.

        Same trio convention as ``pmt``: ordinary annuity (payment at period end), no
        sign flip, ``self`` the per-period rate over ``nper`` like-unit periods. The
        annuity FORM (a payment stream) is the non-redundant complement to
        ``compound`` (36.3), which already grows a lump sum. Closed form
        ``fv = pmt * ((1 + r)**nper - 1) / r``; with a ZERO rate the payments just
        add up, ``pmt * nper``. It COMPUTES by composing +, **, -, * and /, so it
        follows each mode's rule exactly as ``pmt`` does (a whole ``nper`` keeps the
        power exact; the final ``/ r`` may quantize in fixed-point, is exact in
        rational, rounds in float).
        """
        if self._is_zero():
            return pmt.mul(nper)
        one = Value._from_scaled_int(1, 0, self.mode)
        return pmt.mul(one.add(self).pow(nper).sub(one)).div(self)

    def pv(self, nper: "Value", pmt: "Value") -> "Value":
        """PRESENT value of an ``nper``-period stream of level payments ``pmt`` at the
        PER-PERIOD rate ``self`` (36.4) — the ordinary-annuity discount, the inverse
        of ``pmt``.

        Same trio convention (ordinary annuity, no sign flip, per-period rate over
        like-unit ``nper`` periods), the annuity complement to ``compound`` (36.3).
        Closed form ``pv = pmt * (1 - (1 + r)**-nper) / r``; with a ZERO rate the
        undiscounted payments just add up, ``pmt * nper``. It COMPUTES by composing
        +, **, -, * and /, inheriting each mode's rule like the rest of the trio (the
        negative-integer power inverts exactly; the ``/ r`` quantizes in fixed-point,
        is exact in rational, rounds in float).
        """
        if self._is_zero():
            return pmt.mul(nper)
        one = Value._from_scaled_int(1, 0, self.mode)
        return pmt.mul(one.sub(one.add(self).pow(nper.neg()))).div(self)

    def _as_integer(self, what: str) -> int:
        """This Value as a signed Python int, REFUSING any fractional part (40.7).

        The integer-domain gate shared by the integer-only functions (gcd/lcm,
        40.7/40.8): an operand is read as a plain ``int`` in whatever mode
        the run is in, and a non-integer — fixed-point with a fractional part, a
        rational with denominator != 1, a non-whole float — REFUSES with
        ``NotRepresentableError``, the exact-or-refuse stance. The exactness FLAG is
        irrelevant (an inexact ``2.0`` is still the integer 2); only integrality
        matters. The same per-mode test ``_as_ndigits`` (28.22) uses, but kept
        separate: that reads an unsigned-or-signed COUNT, this a value operand, and
        ``what`` names the caller in the refusal message.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if not self.payload.is_integer():
                    raise NotRepresentableError(f"{what} requires integer operands")
                return int(self.payload)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                whole, frac = divmod(fp.mantissa, 10**fp.decimals)
                if frac != 0:
                    raise NotRepresentableError(f"{what} requires integer operands")
                return whole
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload.denominator != 1:
                    raise NotRepresentableError(f"{what} requires integer operands")
                return self.payload.numerator
            case Mode.COMPLEX:
                # gcd/lcm/factorial/comb/perm all read operands through here; complex
                # has no integer ring the engine commits to, so they refuse.
                raise NotRepresentableError(f"{what} is undefined for complex numbers")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def gcd(self, *others: "Value") -> "Value":
        """Greatest common divisor of one-or-more operands (40.7) — VARIADIC; ``math.gcd``.

        The integer-only cousin of ``sum_``/``max_`` (28.5/28.2): ``self`` is the
        first operand, so ``gcd(a)`` is ``|a|`` (the one-operand fold). EXACT in
        EVERY mode — pure integer arithmetic on the operand magnitudes never leaves
        the grid, so no rounding and no inexact flag — and same-mode is enforced
        like every variadic. DOMAIN is integer-valued operands only: each goes
        through ``_as_integer``, so a fixed-point value with a fractional part, a
        rational with denominator != 1, or a non-whole float REFUSES. Sign is
        dropped (``math.gcd`` works on magnitudes) and ``gcd(0, 0)`` is 0. The
        result is a whole number at scale 0 in fixed-point.
        """
        ints = [self._as_integer("gcd")]
        for other in others:
            self._same_mode(other, "gcd")
            ints.append(other._as_integer("gcd"))
        return Value._from_scaled_int(math.gcd(*ints), 0, self.mode)

    def lcm(self, *others: "Value") -> "Value":
        """Least common multiple of one-or-more operands (40.8) — VARIADIC; ``math.lcm``.

        The multiplicative twin of ``gcd`` (40.7): same VARIADIC shape, same
        same-mode enforcement, and the SAME integer-only DOMAIN — each operand goes
        through ``_as_integer``, so a fractional fixed-point value, a rational with
        denominator != 1, or a non-whole float REFUSES. EXACT in EVERY mode: the
        fold ``lcm(a, b) = |a*b| / gcd(a, b)`` stays in the integers, so no rounding
        and no inexact flag. ``lcm(a)`` is ``|a|`` (the one-operand fold), sign is
        dropped (a multiple is taken in magnitude), and ANY zero operand makes the
        whole result 0 (0 shares every multiple), matching ``math.lcm``. The result
        is a whole number at scale 0 in fixed-point.
        """
        ints = [self._as_integer("lcm")]
        for other in others:
            self._same_mode(other, "lcm")
            ints.append(other._as_integer("lcm"))
        return Value._from_scaled_int(math.lcm(*ints), 0, self.mode)

    def factorial(self) -> "Value":
        """n! for a NON-NEGATIVE INTEGER n (40.4) — UNARY, EXACT in every mode (a
        product of integers, no rounding).

        DOMAIN is non-negative integers only: the operand goes through
        ``_as_integer`` (so a fractional fixed-point value, a rational with
        denominator != 1, or a non-whole float REFUSES), and a negative operand
        REFUSES too — both the negative and the non-integer cases are the
        continuous gamma extension n! = gamma(n+1), deferred to 40.4.1. n is capped
        at ``_MAX_FACTORIAL`` so a huge operand cannot lock the process up building
        an astronomically large integer.

        fixed-point / rational: the exact integer ``math.factorial(n)`` at scale 0,
            like gcd/lcm — always exact.
        floating-point: ``float(n!)``, refusing when that overflows a double
            (~n > 170, well under the cap). Marked exact only when the double
            represents n! precisely (every n <= 18; some larger n via n!'s trailing
            factors of two), honoring the exact-in-every-mode contract.
        """
        n = self._as_integer("factorial")
        if n < 0:
            raise NotRepresentableError("factorial of a negative value")
        if n > _MAX_FACTORIAL:
            raise NotRepresentableError(f"factorial argument too large (limit {_MAX_FACTORIAL})")
        f = math.factorial(n)
        match self.mode:
            case Mode.FLOATING_POINT:
                try:
                    fl = float(f)
                except OverflowError:
                    raise NotRepresentableError("factorial overflows floating-point") from None
                return Value(Mode.FLOATING_POINT, fl, exact=(int(fl) == f))
            case Mode.FIXED_POINT | Mode.RATIONAL:
                return Value._from_scaled_int(f, 0, self.mode)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def comb(self, other: "Value") -> "Value":
        """Binomial coefficient C(n, k) = n!/(k!(n-k)!) (40.5) — the count of
        k-element subsets of an n-set. BINARY fixed-arity-2 (the pow/atan2 shape,
        28.20/40.1); ``self`` is n, ``other`` is k.

        DOMAIN integer operands only: both go through ``_as_integer`` (so a
        fractional fixed-point value, a rational with denominator != 1, or a
        non-whole float REFUSES — a non-integer argument is the gamma-generalized
        coefficient, deferred to 40.5.1), with same-mode enforced like every binary
        op. An out-of-range integer k chooses an impossible subset and is 0:
        ``k < 0`` or ``k > n`` (so a negative n, where every k >= 0 already exceeds
        n, also folds to 0). Otherwise EXACT in EVERY mode — ``math.comb`` cancels
        to an integer via the multiplicative form, never three factorials, so no
        rounding. The number of multiplicative terms ``min(k, n-k)`` is capped at
        ``_MAX_FACTORIAL`` so a huge operand cannot lock the process up.

        fixed-point / rational: the exact integer at scale 0, like factorial/gcd.
        floating-point: ``float(C(n, k))``, refusing when that overflows a double
            and marked exact only when the double represents the integer precisely.
        """
        self._same_mode(other, "comb")  # reject mode mixing; exactness is per-mode below
        n = self._as_integer("comb")
        k = other._as_integer("comb")
        if k < 0 or k > n:
            c = 0
        else:
            if min(k, n - k) > _MAX_FACTORIAL:
                raise NotRepresentableError(f"comb argument too large (limit {_MAX_FACTORIAL})")
            c = math.comb(n, k)
        match self.mode:
            case Mode.FLOATING_POINT:
                try:
                    fl = float(c)
                except OverflowError:
                    raise NotRepresentableError("comb overflows floating-point") from None
                return Value(Mode.FLOATING_POINT, fl, exact=(int(fl) == c))
            case Mode.FIXED_POINT | Mode.RATIONAL:
                return Value._from_scaled_int(c, 0, self.mode)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def perm(self, other: "Value") -> "Value":
        """Falling factorial P(n, k) = n!/(n-k)! (40.6) — the count of ordered
        k-permutations of an n-set, comb's order-aware twin. BINARY fixed-arity-2
        (the pow/atan2 shape, 28.20/40.1); ``self`` is n, ``other`` is k.

        DOMAIN integer operands only: both go through ``_as_integer`` (so a
        fractional fixed-point value, a rational with denominator != 1, or a
        non-whole float REFUSES — a non-integer argument is the gamma-generalized
        permutation, deferred to 40.6.1), with same-mode enforced like every binary
        op. An out-of-range integer k arranges an impossible selection and is 0:
        ``k < 0`` or ``k > n`` (so a negative n, where every k >= 0 already exceeds
        n, also folds to 0). Otherwise EXACT in EVERY mode — ``math.perm`` is the
        product of k consecutive integers, so no rounding. The number of
        multiplicative terms ``k`` is capped at ``_MAX_FACTORIAL`` so a huge operand
        cannot lock the process up.

        fixed-point / rational: the exact integer at scale 0, like comb/factorial.
        floating-point: ``float(P(n, k))``, refusing when that overflows a double
            and marked exact only when the double represents the integer precisely.
        """
        self._same_mode(other, "perm")  # reject mode mixing; exactness is per-mode below
        n = self._as_integer("perm")
        k = other._as_integer("perm")
        if k < 0 or k > n:
            p = 0
        else:
            if k > _MAX_FACTORIAL:
                raise NotRepresentableError(f"perm argument too large (limit {_MAX_FACTORIAL})")
            p = math.perm(n, k)
        match self.mode:
            case Mode.FLOATING_POINT:
                try:
                    fl = float(p)
                except OverflowError:
                    raise NotRepresentableError("perm overflows floating-point") from None
                return Value(Mode.FLOATING_POINT, fl, exact=(int(fl) == p))
            case Mode.FIXED_POINT | Mode.RATIONAL:
                return Value._from_scaled_int(p, 0, self.mode)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def sqrt(self) -> "Value":
        """Square root (19.5.2) — irrational, so inexact except where the root
        lands exactly on the mode's own grid. A negative operand has no real
        root and raises NotRepresentableError in every mode (no complex here).

        fixed-point: SUPPORTED (crypto). Take math.isqrt — an exact bignum floor
            — of the mantissa rescaled to the operand's own scale, then round to
            nearest at that scale. A tie cannot occur: sqrt of an integer is never
            a half-integer, so nearest IS half-to-even with nothing to break. The
            result is exact only when the operand was exact AND it is a perfect
            square at that scale (the integer sqrt left no remainder). The target
            scale is the operand's decimals; an optional target-scale arg is still
            to plan (22).
        floating-point: math.sqrt; the result is unconditionally inexact (binary64
            rounds).
        rational: exact ONLY for a perfect square — both numerator and denominator
            are perfect squares; otherwise the root is irrational with no scale to
            round to, so it raises NotRepresentableError.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if self.payload < 0:
                    raise NotRepresentableError("square root of a negative value")
                return Value(Mode.FLOATING_POINT, math.sqrt(self.payload), exact=False)
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa < 0:
                    raise NotRepresentableError("square root of a negative value")
                # sqrt(m * 10**-d) held at scale d == sqrt(m * 10**d) * 10**-d:
                # the integer sqrt of the mantissa rescaled by 10**d, rounded.
                scaled = fp.mantissa * 10**fp.decimals
                root = math.isqrt(scaled)
                remainder = scaled - root * root  # 0 <= remainder <= 2*root
                if remainder > root:  # nearest is root+1 (a tie is impossible)
                    root += 1
                exact = self.exact and remainder == 0
                return Value(Mode.FIXED_POINT, FixedPoint(root, fp.decimals), exact=exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                fr = self.payload
                if fr.numerator < 0:
                    raise NotRepresentableError("square root of a negative value")
                # Fraction is in lowest terms with a positive denominator; the
                # root is rational iff BOTH parts are perfect squares.
                root_num, root_den = math.isqrt(fr.numerator), math.isqrt(fr.denominator)
                if root_num * root_num != fr.numerator or root_den * root_den != fr.denominator:
                    raise NotRepresentableError("rational square root is irrational")
                return Value(Mode.RATIONAL, Fraction(root_num, root_den), exact=self.exact)
            case Mode.COMPLEX:
                return self._complex_sqrt()
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def cbrt(self) -> "Value":
        """Cube root, x**(1/3) (28.21) — the SHAPE of sqrt (19.5.2), inexact except
        where the root lands exactly on the mode's grid. The ONE difference from
        sqrt: a cube root is an ODD root, so NEGATIVE inputs are in DOMAIN —
        cbrt(-8) == -2, real, NOT the NotRepresentableError sqrt raises on negatives.

        fixed-point: SUPPORTED. The integer cube root (the q=3 sibling of math.isqrt,
            _iroot) of the mantissa rescaled to the operand's scale, rounded to
            nearest at that scale. A tie cannot occur (8*scaled is even, (2r+1)**3
            is odd), so nearest is unambiguous. Exact only when the operand was exact
            AND it is a perfect cube at that scale. A negative mantissa cube-roots its
            magnitude and carries the sign.
        floating-point: math.cbrt is 3.11+ but the project floor is 3.10, so compute
            sign-preserving via math.copysign(abs(x)**(1/3), x); unconditionally
            inexact.
        rational: exact ONLY when numerator and denominator are BOTH perfect cubes
            (a negative numerator is fine — odd root); otherwise the root is
            irrational and it raises NotRepresentableError, the exact-or-refuse stance
            of sqrt.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(
                    Mode.FLOATING_POINT,
                    math.copysign(abs(self.payload) ** (1 / 3), self.payload),
                    exact=False,
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                sign = -1 if fp.mantissa < 0 else 1
                # cbrt(m * 10**-d) held at scale d == cbrt(m * 10**(2d)) * 10**-d:
                # the integer cube root of the mantissa rescaled by 10**(2d), rounded.
                scaled = abs(fp.mantissa) * 10 ** (2 * fp.decimals)
                root = _iroot(scaled, 3)
                exact = self.exact and root**3 == scaled
                if 8 * scaled >= (2 * root + 1) ** 3:  # nearest is root+1 (tie impossible)
                    root += 1
                return Value(Mode.FIXED_POINT, FixedPoint(sign * root, fp.decimals), exact=exact)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                fr = self.payload
                # Lowest terms, positive denominator; the root is rational iff BOTH
                # parts are perfect cubes. The numerator may be negative (odd root).
                num_sign = -1 if fr.numerator < 0 else 1
                root_num = _perfect_root(abs(fr.numerator), 3)
                root_den = _perfect_root(fr.denominator, 3)
                if root_num is None or root_den is None:
                    raise NotRepresentableError("rational cube root is irrational")
                return Value(
                    Mode.RATIONAL, Fraction(num_sign * root_num, root_den), exact=self.exact
                )
            case Mode.COMPLEX:
                raise NotRepresentableError("cbrt is not supported for complex numbers")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def sin(self) -> "Value":
        """Sine, argument in radians (28.10) — transcendental, so inexact except
        the trivial sin(0) = 0.

        fixed-point: SUPPORTED. Range-reduce the argument mod 2*pi into (-pi, pi]
            using the internal pi helper (28.10.1), then sum the Taylor series
            ``x - x**3/3! + x**5/5! - ...`` on scaled ints at the operand's scale
            plus guard digits, rounding half-to-even back to that scale. The result
            is ALWAYS inexact (transcendental) except sin(0) = 0, which is exact.
            The reduced argument is folded to non-negative first (sin is odd): the
            integer term recurrence floors toward -inf, so a negative argument would
            stick the alternating series at -1 forever instead of converging to 0.
        floating-point: math.sin; unconditionally inexact.
        rational: sin of a rational is irrational except sin(0) = 0; otherwise there
            is no scale to round to, so it raises NotRepresentableError (the same
            exact-or-refuse stance as sqrt).
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.sin(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_sin()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("sine of a non-zero rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # sin(0) = 0, the only exact fixed-point case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                reduced, working = _fp_reduce_mod_2pi(fp)
                negate = reduced < 0  # sin is odd; sum the series on |reduced|
                total = _fp_sin_series(abs(reduced), 10**working)
                if negate:
                    total = -total
                # Round the working-scale total back to the operand's scale.
                mantissa, _ = _fp_quantize(total, 10 ** (working - fp.decimals), 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def cos(self) -> "Value":
        """Cosine, argument in radians (28.11) — the SAME machinery as sin (28.10),
        transcendental, so inexact except the trivial cos(0) = 1.

        fixed-point: SUPPORTED. Range-reduce mod 2*pi via the shared reducer, then
            sum the Taylor series ``1 - x**2/2! + x**4/4! - ...`` on scaled ints at
            the operand's scale plus guard digits, rounding half-to-even back. cos
            is EVEN, so the series runs on |reduced| with no sign to restore (and
            the non-negative argument keeps the integer recurrence terminating).
            Inexact except cos(0) = 1, which is exact.
        floating-point: math.cos; unconditionally inexact.
        rational: cos of a rational is irrational except cos(0) = 1; otherwise there
            is no scale to round to, so it raises NotRepresentableError.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.cos(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_cos()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:
                    return Value(Mode.RATIONAL, Fraction(1), exact=self.exact)
                raise NotRepresentableError("cosine of a non-zero rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # cos(0) = 1, the only exact fixed-point case
                    one = 10**fp.decimals  # 1.000...0 as a mantissa at this scale
                    return Value(Mode.FIXED_POINT, FixedPoint(one, fp.decimals), exact=self.exact)
                reduced, working = _fp_reduce_mod_2pi(fp)
                total = _fp_cos_series(abs(reduced), 10**working)  # cos is even; no sign
                # Round the working-scale total back to the operand's scale.
                mantissa, _ = _fp_quantize(total, 10 ** (working - fp.decimals), 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def tan(self) -> "Value":
        """Tangent, argument in radians (28.12) — == sin/cos (28.10/28.11),
        transcendental, so inexact except the trivial tan(0) = 0.

        fixed-point: SUPPORTED. Range-reduce mod 2*pi once via the shared reducer,
            then sum BOTH the sine and cosine Taylor series on scaled ints at the
            working scale and divide — the scale cancels in the ratio, and the
            result is rounded half-to-even to the operand's scale. tan is undefined
            at odd multiples of pi/2 where cos = 0; pi is irrational so that point
            is never hit exactly (the answer there is just a large inexact value),
            UNLESS the cosine series rounds to 0 at the working scale, which raises
            ZeroDivisionError. Inexact except tan(0) = 0, which is exact.
        floating-point: math.tan; unconditionally inexact.
        rational: tan of a rational is irrational except tan(0) = 0; otherwise there
            is no scale to round to, so it raises NotRepresentableError.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.tan(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_tan()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("tangent of a non-zero rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # tan(0) = 0, the only exact fixed-point case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                reduced, working = _fp_reduce_mod_2pi(fp)
                unity = 10**working
                sin_total = _fp_sin_series(abs(reduced), unity)
                if reduced < 0:  # sin is odd (cos, summed on |reduced|, is even)
                    sin_total = -sin_total
                cos_total = _fp_cos_series(abs(reduced), unity)
                if cos_total == 0:  # cos rounded to 0 at the working scale -> undefined
                    raise ZeroDivisionError("fixed-point tangent of an odd multiple of pi/2")
                # tan == sin/cos; the unity scale cancels, so quantize the bare ratio.
                mantissa, _ = _fp_quantize(sin_total * 10**fp.decimals, cos_total, 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def cot(self) -> "Value":
        """Cotangent, argument in radians (28.13) — == cos/sin, the mirror of tan
        (28.12), transcendental and inexact everywhere it is defined.

        fixed-point: SUPPORTED. Range-reduce mod 2*pi once via the shared reducer,
            then sum BOTH the cosine and sine Taylor series on scaled ints at the
            working scale and divide — the scale cancels in the ratio, rounded
            half-to-even to the operand's scale. cot is undefined at multiples of pi
            where sin = 0; pi is irrational so that point is never hit exactly (the
            answer there is just a large inexact value), UNLESS the sine series
            rounds to 0 at the working scale, which raises ZeroDivisionError. The
            argument 0 IS sin = 0 exactly, so cot(0) is undefined — there is no
            trivial exact case (the mirror of tan(0) = 0). Always inexact.
        floating-point: math.cos / math.sin; unconditionally inexact, and a zero
            sine (cot at a multiple of pi) propagates Python's ZeroDivisionError.
        rational: cot of a rational is irrational, so it raises NotRepresentableError
            — except the argument 0, where sin = 0 makes it undefined (ZeroDivisionError).
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                cot = math.cos(self.payload) / math.sin(self.payload)
                return Value(Mode.FLOATING_POINT, cot, exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("cot is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:
                    raise ZeroDivisionError("cotangent of zero is undefined")
                raise NotRepresentableError("cotangent of a non-zero rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # cot(0) = cos/sin = 1/0 — undefined (sin = 0)
                    raise ZeroDivisionError("fixed-point cotangent of a multiple of pi")
                reduced, working = _fp_reduce_mod_2pi(fp)
                unity = 10**working
                sin_total = _fp_sin_series(abs(reduced), unity)
                if reduced < 0:  # sin is odd (cos, summed on |reduced|, is even)
                    sin_total = -sin_total
                if sin_total == 0:  # sin rounded to 0 at the working scale -> undefined
                    raise ZeroDivisionError("fixed-point cotangent of a multiple of pi")
                cos_total = _fp_cos_series(abs(reduced), unity)
                # cot == cos/sin; the unity scale cancels, so quantize the bare ratio.
                mantissa, _ = _fp_quantize(cos_total * 10**fp.decimals, sin_total, 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def asin(self) -> "Value":
        """Arcsine, result in radians within [-pi/2, pi/2] (28.14) — transcendental,
        so inexact except the trivial asin(0) = 0. DOMAIN-RESTRICTED to |x| <= 1: an
        argument outside [-1, 1] has no real arcsine and raises NotRepresentableError
        in every mode (the domain refusal mirrors sqrt's on a negative operand).

        fixed-point: SUPPORTED. ``asin(x) == atan(x / sqrt(1 - x**2))`` — the plain
            arcsine series converges badly near |x| = 1, so it routes through an
            internal Machin-family arctan series (the unit-fraction ``_arctan_inv``
            behind pi, lifted to a general argument, with one half-angle reduction
            for fast convergence) reusing the fixed-point sqrt, at the operand's
            scale plus guard digits and rounded half-to-even back. Always inexact
            except asin(0) = 0, which is exact.
        floating-point: math.asin; unconditionally inexact.
        rational: asin of a rational is irrational except asin(0) = 0; an in-domain
            non-zero argument therefore raises NotRepresentableError, the same
            exact-or-refuse stance as sqrt/sin.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if abs(self.payload) > 1:
                    raise NotRepresentableError("arcsine argument outside the domain [-1, 1]")
                return Value(Mode.FLOATING_POINT, math.asin(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("asin is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if abs(self.payload) > 1:
                    raise NotRepresentableError("arcsine argument outside the domain [-1, 1]")
                if self.payload == 0:  # asin(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("arcsine of a non-zero rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if abs(fp.mantissa) > 10**fp.decimals:  # |x| > 1 -> outside the domain
                    raise NotRepresentableError("arcsine argument outside the domain [-1, 1]")
                if fp.mantissa == 0:  # asin(0) = 0, the only exact fixed-point case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                sign = -1 if fp.mantissa < 0 else 1  # asin is odd; compute on |x|
                magnitude = _fp_asin(abs(fp.mantissa), fp.decimals)
                result = FixedPoint(sign * magnitude, fp.decimals)
                return Value(Mode.FIXED_POINT, result, exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def acos(self) -> "Value":
        """Arccosine, result in radians within [0, pi] (28.15) — transcendental, so
        inexact except the trivial acos(1) = 0. DOMAIN-RESTRICTED to |x| <= 1, like
        asin (28.14): an argument outside [-1, 1] raises NotRepresentableError in
        every mode.

        ``acos(x) == pi/2 - asin(x)``, so it reuses asin's arctan machinery and the
        internal pi helper (28.10.1) for pi/2; the per-mode story is asin's.
        fixed-point: SUPPORTED. Subtract the un-rounded working-scale asin from pi/2
            and round once. Always inexact except acos(1) = 0, which is exact (asin(1)
            = pi/2 cancels exactly). NB acos's exact landmark is x = 1, NOT x = 0
            (acos(0) = pi/2 is irrational) — the mirror of asin's.
        floating-point: math.acos; unconditionally inexact.
        rational: acos of a rational is irrational except acos(1) = 0; any other
            in-domain argument raises NotRepresentableError.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if abs(self.payload) > 1:
                    raise NotRepresentableError("arccosine argument outside the domain [-1, 1]")
                return Value(Mode.FLOATING_POINT, math.acos(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("acos is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if abs(self.payload) > 1:
                    raise NotRepresentableError("arccosine argument outside the domain [-1, 1]")
                if self.payload == 1:  # acos(1) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("arccosine of a rational other than 1 is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if abs(fp.mantissa) > 10**fp.decimals:  # |x| > 1 -> outside the domain
                    raise NotRepresentableError("arccosine argument outside the domain [-1, 1]")
                if fp.mantissa == 10**fp.decimals:  # acos(1) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                asin_scaled, working = _fp_asin_scaled(abs(fp.mantissa), fp.decimals)
                if fp.mantissa < 0:  # asin is odd; acos(-x) = pi/2 + asin(|x|)
                    asin_scaled = -asin_scaled
                acos_scaled = _pi_scaled(working) // 2 - asin_scaled
                mantissa, _ = _fp_quantize(acos_scaled, 10 ** (working - fp.decimals), 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def atan(self) -> "Value":
        """Arctangent, result in radians within (-pi/2, pi/2) (28.16) — transcendental,
        so inexact except the trivial atan(0) = 0. UNRESTRICTED domain: every real has
        an arctangent, so unlike asin/acos (28.14/28.15) no argument is ever refused
        for being out of domain.

        atan is the primitive the inverse trig reduces to — the Machin-type arctan
        series behind pi (28.10.1), lifted to a general argument for asin (28.14), IS
        atan, so exposing it is nearly free.
        fixed-point: SUPPORTED. The internal arctan series at the operand's scale plus
            guard digits, rounded half-to-even back; arguments above 1 reduce with
            ``atan(x) == pi/2 - atan(1/x)``. Always inexact except atan(0) = 0.
        floating-point: math.atan; unconditionally inexact.
        rational: atan of a rational is irrational except atan(0) = 0; a non-zero
            argument therefore raises NotRepresentableError, the exact-or-refuse stance
            shared with asin/sin.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.atan(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("atan is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # atan(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("arctangent of a non-zero rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # atan(0) = 0, the only exact fixed-point case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                sign = -1 if fp.mantissa < 0 else 1  # atan is odd; compute on |x|
                magnitude = _fp_atan(abs(fp.mantissa), fp.decimals)
                result = FixedPoint(sign * magnitude, fp.decimals)
                return Value(Mode.FIXED_POINT, result, exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def atan2(self, other: "Value") -> "Value":
        """Two-argument arctangent ``atan2(y, x)`` — the angle in radians of the point
        ``(x, y)`` measured from the positive x-axis, in ``(-pi, pi]`` (40.1). Unlike
        plain ``atan(y/x)`` (28.16) it knows the QUADRANT from the two signs, so it
        distinguishes ``(x, y)`` from ``(-x, -y)`` and is defined when ``x == 0``.
        ``self`` is ``y``, ``other`` is ``x`` (the math.atan2 argument order). The
        BINARY fixed-arity-2 shape (pow, 28.20). Transcendental, so inexact except the
        single exact landmark ``atan2(0, x>=0) == 0`` (which, by the math.atan2
        convention, also fixes ``atan2(0, 0) == 0``).

        fixed-point: SUPPORTED. The base angle ``atan(|y/x|)`` is the arctan series at
            ``max scale + _PI_GUARD`` guard digits — the ratio fed in directly as
            ``|y|/|x|`` so no division rounds first — then offset by ``pi``/``pi/2``
            (the internal pi, 28.10.1) for the quadrant and rounded half-to-even back
            to the covering scale. Always inexact except the zero landmark.
        floating-point: math.atan2; unconditionally inexact.
        rational: irrational except ``atan2(0, x>=0) == 0`` (any other point's angle is
            a non-zero multiple/sum involving pi or an irrational arctan); otherwise
            NotRepresentableError, the exact-or-refuse stance shared with atan.
        """
        exact = self._same_mode(other, "atan2")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(other.payload, float)
                return Value(
                    Mode.FLOATING_POINT, math.atan2(self.payload, other.payload), exact=False
                )
            case Mode.COMPLEX:
                raise NotRepresentableError("atan2 is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(other.payload, Fraction)
                if (
                    self.payload == 0 and other.payload >= 0
                ):  # atan2(0, x>=0) = 0, the only exact case
                    return Value(Mode.RATIONAL, Fraction(0), exact=exact)
                raise NotRepresentableError(
                    "angle of a rational point is irrational except atan2(0, x>=0) = 0"
                )
            case Mode.FIXED_POINT:
                a, b = self._fp_pair(other)  # a = y, b = x
                scale = max(a.decimals, b.decimals)  # the covering result scale (19.1.2)
                if a.mantissa == 0 and b.mantissa >= 0:  # atan2(0, x>=0) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, scale), exact=exact)
                working = scale + _PI_GUARD
                unity = 10**working
                if a.mantissa == 0:  # y = 0, x < 0: theta = pi
                    theta = _pi_scaled(working)
                elif b.mantissa == 0:  # x = 0, y != 0: theta = +/- pi/2
                    theta = _pi_scaled(working) // 2
                    if a.mantissa < 0:
                        theta = -theta
                else:
                    # base = atan(|y/x|) in [0, pi/2], the ratio fed in un-rounded.
                    ny = abs(a.mantissa) * 10**b.decimals  # |y| and |x| at a common scale
                    nx = abs(b.mantissa) * 10**a.decimals
                    if ny <= nx:  # |y/x| <= 1: the arctan series argument is in range
                        base = _fp_arctan_series(ny * unity // nx, unity)
                    else:  # |y/x| > 1: atan(t) = pi/2 - atan(1/t), with 1/t = |x/y| <= 1
                        base = _pi_scaled(working) // 2 - _fp_arctan_series(nx * unity // ny, unity)
                    if b.mantissa > 0:  # quadrants I/IV: theta = +/- base
                        theta = base if a.mantissa > 0 else -base
                    else:  # x < 0, quadrants II/III: theta = +/-(pi - base)
                        pi = _pi_scaled(working)
                        theta = pi - base if a.mantissa > 0 else base - pi
                mantissa, _ = _fp_quantize(theta, 10 ** (working - scale), 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def degrees(self) -> "Value":
        """Convert an angle from radians to degrees: ``x * 180/pi`` (40.11). UNIT
        scaling, NOT a trig function — every real converts, no domain limit (the
        bridge OUT of the radians-only trig family, 28.10). Multiplies by an
        irrational (1/pi), so inexact except the trivial degrees(0) = 0.

        fixed-point: SUPPORTED via the internal pi (29.3/28.10.1). Form the exact
            ratio ``m * 180 * 10**working / (pi_scaled(working) * 10**decimals)`` and
            round half-to-even to the operand's scale; ``working`` carries the
            argument's integer digits plus guard so pi's last place never reaches the
            result. Inexact except degrees(0) = 0, which is exact.
        floating-point: math.degrees; unconditionally inexact.
        rational: pi has no rational value, so degrees(x) is irrational except
            degrees(0) = 0; otherwise NotRepresentableError, the exact-or-refuse stance.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.degrees(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("degrees is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # degrees(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("degrees of a non-zero rational is irrational (pi)")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # degrees(0) = 0, the only exact fixed-point case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                int_digits = len(str(abs(fp.mantissa) // 10**fp.decimals))
                working = fp.decimals + int_digits + _PI_GUARD
                pi = _pi_scaled(working)
                # x * 180/pi, the d's cancel: mantissa = round(m * 180 / pi).
                num = fp.mantissa * 180 * 10**working
                den = pi * 10**fp.decimals
                mantissa, _ = _fp_quantize(num, den, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def radians(self) -> "Value":
        """Convert an angle from degrees to radians: ``x * pi/180`` (40.11) — the
        inverse of ``degrees`` (the bridge INTO the radians-only trig family, 28.10).
        UNIT scaling, NOT a trig function: every real converts, no domain limit.
        Multiplies by an irrational (pi), so inexact except the trivial radians(0) = 0.

        fixed-point: SUPPORTED via the internal pi (29.3/28.10.1). Form the exact
            ratio ``m * pi_scaled(working) / (180 * 10**(decimals + working))`` and
            round half-to-even to the operand's scale; ``working`` carries the
            argument's integer digits plus guard so pi's last place never reaches the
            result. Inexact except radians(0) = 0, which is exact.
        floating-point: math.radians; unconditionally inexact.
        rational: pi has no rational value, so radians(x) is irrational except
            radians(0) = 0; otherwise NotRepresentableError, the exact-or-refuse stance.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.radians(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("radians is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # radians(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("radians of a non-zero rational is irrational (pi)")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # radians(0) = 0, the only exact fixed-point case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                int_digits = len(str(abs(fp.mantissa) // 10**fp.decimals))
                working = fp.decimals + int_digits + _PI_GUARD
                pi = _pi_scaled(working)
                # x * pi/180, the d's cancel: mantissa = round(m * pi / 180).
                num = fp.mantissa * pi
                den = 180 * 10 ** (fp.decimals + working)
                mantissa, _ = _fp_quantize(num, den, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def log(self, base: "Value | None" = None) -> "Value":
        """NATURAL logarithm base e when unary (28.17; bare ``log`` is ln, with the
        ``ln`` alias); the GENERAL logarithm ``log(x)/log(base)`` when a second
        operand supplies the base (40.10). Transcendental, so inexact except the
        trivial log(1) = 0. DOMAIN x > 0: a non-positive operand (x = 0 is -inf,
        x < 0 undefined) raises NotRepresentableError in every mode, mirroring
        sqrt's negative refusal.

        The PRIMITIVE the log family reduces to: log10/log2 (28.18/28.19) are this
        divided by the ln(10)/ln(2) constant, and the two-arg ``log(x, base)``
        (``_log_base``) is the ratio of two of these.

        fixed-point: SUPPORTED via base-10 reduction + an atanh series (the _fp_ln
            core, 28.17): ``x == r*10**n`` so ``ln(x) == n*ln(10) + 2*atanh((r-1)/
            (r+1))``, summed on scaled ints at the operand's scale plus guard digits
            and rounded half-to-even back. Transcendental -> always inexact except
            log(1) = 0, which is exact.
        floating-point: math.log; unconditionally inexact.
        rational: log of a rational is transcendental except log(1) = 0 (Lindemann-
            Weierstrass); otherwise NotRepresentableError, the exact-or-refuse stance
            of sqrt.
        """
        if self.mode is Mode.COMPLEX:
            # complex log handles its own (optional) base, so it intercepts before the
            # real two-arg path; ln(z) = ln|z| + arg(z)*i, refusing z == 0.
            return self._complex_log(base)
        if base is not None:
            return self._log_base(base)
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                return Value(Mode.FLOATING_POINT, math.log(self.payload), exact=False)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if self.payload == 1:  # log(1) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("logarithm of a non-unit rational is transcendental")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if fp.mantissa == 10**fp.decimals:  # log(1) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                ln_total, working = _fp_ln(fp)
                # Round the working-scale natural log back to the operand's scale.
                mantissa, _ = _fp_quantize(ln_total, 10 ** (working - fp.decimals), 0)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def ln(self) -> "Value":
        """Natural logarithm base e — the ``ln`` spelling of unary ``log`` (28.17).

        Kept a strictly UNARY method on purpose: ``log`` overloads to ``log(x, base)``
        (40.10, arity (1, 2)), but ``ln`` means natural-log-only, so its arity stays
        (1, 1). A thin wrapper, not a registry alias of ``log``, for exactly that.
        """
        return self.log()

    def _log_base(self, base: "Value") -> "Value":
        """General logarithm ``log(x)/log(base)`` — the two-arg ``log(x, base)`` (40.10).

        DOMAIN x > 0 AND base > 0, base != 1 (a non-positive or unit base has no
        logarithm); refused in every mode like the unary log's x <= 0. EXACT only
        when x is an integer power of base — log(8, 2) = 3, log(1, base) = 0,
        log(base, base) = 1 (``_integer_log``) — which lands the whole exponent with
        no rounding; every other case is transcendental.

        fixed-point: the integer-power landmark returns that exponent verbatim;
            otherwise the ratio of two _fp_ln cores (28.17), whose working scales
            cancel in the quotient, quantized half-to-even to the wider operand's
            scale; inexact.
        floating-point: math.log(x, base); unconditionally inexact, even on a landmark.
        rational: exact only on an integer-power landmark (the result exponent);
            otherwise transcendental -> NotRepresentableError, the exact-or-refuse
            stance of the unary log.
        """
        exact = self._same_mode(base, "log")
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float) and isinstance(base.payload, float)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if base.payload <= 0 or base.payload == 1:
                    raise NotRepresentableError("logarithm base must be positive and not 1")
                return Value(Mode.FLOATING_POINT, math.log(self.payload, base.payload), exact=False)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction) and isinstance(base.payload, Fraction)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if base.payload <= 0 or base.payload == 1:
                    raise NotRepresentableError("logarithm base must be positive and not 1")
                k = _integer_log(self.payload, base.payload)
                if k is not None:
                    return Value(Mode.RATIONAL, Fraction(k), exact=exact)
                raise NotRepresentableError("logarithm with a non-power base is transcendental")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint) and isinstance(base.payload, FixedPoint)
                x_fp, b_fp = self.payload, base.payload
                if x_fp.mantissa <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if b_fp.mantissa <= 0 or b_fp.mantissa == 10**b_fp.decimals:
                    raise NotRepresentableError("logarithm base must be positive and not 1")
                scale = max(x_fp.decimals, b_fp.decimals)
                x_fr = Fraction(x_fp.mantissa, 10**x_fp.decimals)
                b_fr = Fraction(b_fp.mantissa, 10**b_fp.decimals)
                k = _integer_log(x_fr, b_fr)
                if k is not None:
                    return Value(Mode.FIXED_POINT, FixedPoint(k * 10**scale, scale), exact=exact)
                ln_x, wx = _fp_ln(x_fp)
                ln_b, wb = _fp_ln(b_fp)
                # log_base(x) == ln(x)/ln(base) == (ln_x/10**wx)/(ln_b/10**wb)
                #             == (ln_x * 10**wb) / (ln_b * 10**wx); the scales cancel.
                mantissa, _ = _fp_quantize(ln_x * 10**wb, ln_b * 10**wx, scale)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, scale), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def log10(self) -> "Value":
        """BASE-10 logarithm == log(x) / ln(10) (28.18) — transcendental like log
        (28.17), but with RICHER exact landmarks: the base-10 reduction already
        splits ``x == r * 10**n``, so ``log10(x) == n + log(r)/ln(10)`` and a power
        of ten (r = 1) logs to the whole number ``n`` EXACTLY. DOMAIN x > 0; a
        non-positive operand raises NotRepresentableError in every mode, like log.

        fixed-point: SUPPORTED. A power of ten at the operand's scale (mantissa ==
            10**j) is the exponent ``j - decimals`` exactly — e.g. log10(100) = 2,
            log10(0.001) = -3. Otherwise reuse the _fp_ln core and divide by the
            ln(10) constant at the working scale (the scale cancels in the ratio),
            quantizing half-to-even to the operand's scale; inexact.
        floating-point: math.log10; unconditionally inexact.
        rational: log10 of a rational is irrational UNLESS x is an integer power of
            ten (x == 10**k, k possibly negative like 1/10), where it is ``k``
            exactly; otherwise NotRepresentableError. A cleaner exact set than
            sqrt's perfect-square — the log analogue of it.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                return Value(Mode.FLOATING_POINT, math.log10(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_log_int_base(10)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                fr = self.payload
                if fr <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                # x == 10**k is the only rational-result case: 10**k for k >= 0 is
                # numerator/1, for k < 0 it is 1/denominator (both in lowest terms).
                if fr.denominator == 1 and (j := _power_of_ten_exponent(fr.numerator)) is not None:
                    return Value(Mode.RATIONAL, Fraction(j), exact=self.exact)
                if fr.numerator == 1 and (j := _power_of_ten_exponent(fr.denominator)) is not None:
                    return Value(Mode.RATIONAL, Fraction(-j), exact=self.exact)
                raise NotRepresentableError(
                    "base-10 logarithm of a non-power-of-ten rational is irrational"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                # x == 10**(j - decimals) is exact: the result is that whole exponent.
                if (j := _power_of_ten_exponent(fp.mantissa)) is not None:
                    exponent = j - fp.decimals
                    return Value(
                        Mode.FIXED_POINT,
                        FixedPoint(exponent * 10**fp.decimals, fp.decimals),
                        exact=self.exact,
                    )
                ln_total, working = _fp_ln(fp)
                # log10 == ln(x) / ln(10); the working scale cancels in the ratio,
                # so quantize the bare quotient to the operand's scale.
                mantissa, _ = _fp_quantize(ln_total, _ln10_scaled(working), fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def log2(self) -> "Value":
        """BASE-2 logarithm == log(x) / ln(2) (28.19) — transcendental like log
        (28.17), planned with the log family because this engine is bit-oriented
        (entropy / bit-width work). DOMAIN x > 0; a non-positive operand raises
        NotRepresentableError in every mode, like log.

        Unlike log10 the argument reduction is base 10, NOT base 2, so a power of
        two does NOT fall out exactly — the ONLY exact landmark is the trivial
        log2(1) = 0. (A base-2 reduction would land powers of two exactly but
        breaks the shared base-10 ln core; not worth a second reduction path.)

        fixed-point: SUPPORTED. Reuse the _fp_ln core and divide by the ln(2)
            constant at the working scale (the scale cancels in the ratio),
            quantizing half-to-even to the operand's scale; inexact except
            log2(1) = 0.
        floating-point: math.log2; unconditionally inexact.
        rational: log2 of a rational is irrational except log2(1) = 0; otherwise
            NotRepresentableError, the exact-or-refuse stance of log.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                return Value(Mode.FLOATING_POINT, math.log2(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_log_int_base(2)
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if self.payload == 1:  # log2(1) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError("base-2 logarithm of a non-unit rational is irrational")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa <= 0:
                    raise NotRepresentableError("logarithm of a non-positive value")
                if fp.mantissa == 10**fp.decimals:  # log2(1) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                ln_total, working = _fp_ln(fp)
                # log2 == ln(x) / ln(2); the working scale cancels in the ratio,
                # so quantize the bare quotient to the operand's scale.
                mantissa, _ = _fp_quantize(ln_total, _ln2_scaled(working), fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def exp(self) -> "Value":
        """The exponential e**x (28.27), the INVERSE of log/ln (28.17) —
        transcendental, so inexact except the trivial exp(0) = 1.

        fixed-point: SUPPORTED via the all-plus Taylor series 1 + x + x**2/2! + ...
            (the _fp_exp_series core, 28.27.1) after range-reducing by ln(2): write
            x == k*ln2 + s with s in [0, ln 2), so exp(x) == 2**k * exp(s) — the
            2**k an EXACT mantissa shift, only exp(s) summed. The working scale
            carries the result's ~k*log10(2) integer digits on top of the operand's
            so the shift stays accurate to the last place. Negative x is fine
            (k < 0). Always inexact except exp(0) = 1.
        floating-point: math.exp; unconditionally inexact.
        rational: exp of a rational is transcendental except exp(0) = 1 (Lindemann-
            Weierstrass: e**r is irrational for rational r != 0); otherwise
            NotRepresentableError, the exact-or-refuse stance of log/sqrt.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.exp(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_exp()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # exp(0) = 1, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(1), exact=self.exact)
                raise NotRepresentableError("exp of a non-zero rational is transcendental")
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # exp(0) = 1, the only exact case
                    return Value(
                        Mode.FIXED_POINT, FixedPoint(10**fp.decimals, fp.decimals), exact=self.exact
                    )
                ratio = _fp_exp_ratio(fp)  # the un-rounded e**x core (28.27.1)
                mantissa, _ = _fp_quantize(ratio.numerator, ratio.denominator, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def sinh(self) -> "Value":
        """Hyperbolic sine ``(e**x - e**-x)/2`` (40.2) — transcendental like exp
        (28.27), so inexact except the trivial sinh(0) = 0.

        fixed-point: SUPPORTED via the exp core (_fp_exp_pm): combine the un-rounded
            e**x and e**-x and round the half-difference ONCE to the operand's scale.
            Always inexact except sinh(0) = 0, which is exact.
        floating-point: math.sinh; unconditionally inexact.
        rational: sinh of a rational is transcendental except sinh(0) = 0 (it is built
            from exp, irrational at every non-zero rational); otherwise
            NotRepresentableError, the exact-or-refuse stance of exp.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.sinh(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_sinh()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # sinh(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError(
                    "hyperbolic sine of a non-zero rational is transcendental"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # sinh(0) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                ex, emx = _fp_exp_pm(fp)
                r = (ex - emx) / 2
                mantissa, _ = _fp_quantize(r.numerator, r.denominator, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def cosh(self) -> "Value":
        """Hyperbolic cosine ``(e**x + e**-x)/2`` (40.2) — transcendental like exp
        (28.27), so inexact except the trivial cosh(0) = 1 (the only exact landmark;
        cosh's mirror of sinh/tanh's zero, since cosh is even with a minimum of 1).

        fixed-point: SUPPORTED via the exp core (_fp_exp_pm): round the half-SUM of the
            un-rounded e**x and e**-x once. Always inexact except cosh(0) = 1.
        floating-point: math.cosh; unconditionally inexact.
        rational: transcendental except cosh(0) = 1; otherwise NotRepresentableError,
            the exact-or-refuse stance of exp.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.cosh(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_cosh()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # cosh(0) = 1, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(1), exact=self.exact)
                raise NotRepresentableError(
                    "hyperbolic cosine of a non-zero rational is transcendental"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # cosh(0) = 1, the only exact case
                    return Value(
                        Mode.FIXED_POINT, FixedPoint(10**fp.decimals, fp.decimals), exact=self.exact
                    )
                ex, emx = _fp_exp_pm(fp)
                r = (ex + emx) / 2
                mantissa, _ = _fp_quantize(r.numerator, r.denominator, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def tanh(self) -> "Value":
        """Hyperbolic tangent ``(e**x - e**-x)/(e**x + e**-x)`` (40.2) — transcendental
        like exp (28.27), so inexact except the trivial tanh(0) = 0. The denominator
        cosh is never zero, so tanh has NO domain restriction (range (-1, 1)).

        fixed-point: SUPPORTED via the exp core (_fp_exp_pm): round the ratio of the
            un-rounded e**x - e**-x and e**x + e**-x once. Always inexact except
            tanh(0) = 0.
        floating-point: math.tanh; unconditionally inexact.
        rational: transcendental except tanh(0) = 0; otherwise NotRepresentableError,
            the exact-or-refuse stance of exp.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.tanh(self.payload), exact=False)
            case Mode.COMPLEX:
                return self._complex_tanh()
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # tanh(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError(
                    "hyperbolic tangent of a non-zero rational is transcendental"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # tanh(0) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                ex, emx = _fp_exp_pm(fp)
                r = (ex - emx) / (ex + emx)  # cosh > 0, so the denominator never vanishes
                mantissa, _ = _fp_quantize(r.numerator, r.denominator, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(mantissa, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def asinh(self) -> "Value":
        """Inverse hyperbolic sine ``ln(x + sqrt(x**2 + 1))`` (40.3) — the inverse of
        sinh (40.2), transcendental like log (28.17), so inexact except asinh(0) = 0.
        UNRESTRICTED domain: every real has an inverse hyperbolic sine, so (like atan,
        28.16) no argument is ever refused.

        fixed-point: SUPPORTED via the ln core (_fp_asinh). asinh is odd, so the sign is
            folded off and restored — the magnitude runs on |x|, where ``x + sqrt(...)
            >= 1`` avoids cancellation. Always inexact except asinh(0) = 0.
        floating-point: math.asinh; unconditionally inexact.
        rational: transcendental except asinh(0) = 0 (it reduces to a logarithm);
            otherwise NotRepresentableError, the exact-or-refuse stance of log.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return Value(Mode.FLOATING_POINT, math.asinh(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("asinh is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload == 0:  # asinh(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError(
                    "inverse hyperbolic sine of a non-zero rational is transcendental"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa == 0:  # asinh(0) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                sign = -1 if fp.mantissa < 0 else 1  # asinh is odd; compute on |x|
                magnitude = _fp_asinh(abs(fp.mantissa), fp.decimals)
                return Value(
                    Mode.FIXED_POINT, FixedPoint(sign * magnitude, fp.decimals), exact=False
                )
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def acosh(self) -> "Value":
        """Inverse hyperbolic cosine ``ln(x + sqrt(x**2 - 1))`` (40.3) — the inverse of
        cosh (40.2), transcendental like log (28.17), so inexact except acosh(1) = 0.
        DOMAIN x >= 1 (cosh's range): an argument below 1 raises NotRepresentableError in
        every mode, like sqrt's negative refusal.

        fixed-point: SUPPORTED via the ln core (_fp_acosh) for x > 1; always inexact
            except acosh(1) = 0, which is exact (the radicand vanishes, ln(1) = 0).
        floating-point: math.acosh; unconditionally inexact.
        rational: acosh of a rational is irrational except acosh(1) = 0; any other
            in-domain argument raises NotRepresentableError.
        """
        below = "inverse hyperbolic cosine argument below the domain [1, inf)"
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if self.payload < 1:
                    raise NotRepresentableError(below)
                return Value(Mode.FLOATING_POINT, math.acosh(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("acosh is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload < 1:
                    raise NotRepresentableError(below)
                if self.payload == 1:  # acosh(1) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError(
                    "inverse hyperbolic cosine of a rational above 1 is transcendental"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if fp.mantissa < 10**fp.decimals:  # x < 1 -> below the domain
                    raise NotRepresentableError(below)
                if fp.mantissa == 10**fp.decimals:  # acosh(1) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                magnitude = _fp_acosh(fp.mantissa, fp.decimals)
                return Value(Mode.FIXED_POINT, FixedPoint(magnitude, fp.decimals), exact=False)
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def atanh(self) -> "Value":
        """Inverse hyperbolic tangent ``ln((1 + x)/(1 - x))/2`` (40.3) — the inverse of
        tanh (40.2), transcendental like log (28.17), so inexact except atanh(0) = 0.
        DOMAIN |x| < 1 (tanh's open range); x = +/-1 is +/-inf, so an argument with
        |x| >= 1 raises NotRepresentableError in every mode.

        fixed-point: SUPPORTED via the ln core (_fp_atanh) — through the public ln, NOT
            the bare atanh series (which is ln's own internal core, 28.17.1). atanh is
            odd, so the sign is folded off and restored. Always inexact except
            atanh(0) = 0.
        floating-point: math.atanh; unconditionally inexact.
        rational: transcendental except atanh(0) = 0; any other in-domain argument
            raises NotRepresentableError, the exact-or-refuse stance of log.
        """
        outside = "inverse hyperbolic tangent argument outside the domain (-1, 1)"
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                if abs(self.payload) >= 1:
                    raise NotRepresentableError(outside)
                return Value(Mode.FLOATING_POINT, math.atanh(self.payload), exact=False)
            case Mode.COMPLEX:
                raise NotRepresentableError("atanh is not supported for complex numbers")
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if abs(self.payload) >= 1:
                    raise NotRepresentableError(outside)
                if self.payload == 0:  # atanh(0) = 0, the only exact rational case
                    return Value(Mode.RATIONAL, Fraction(0), exact=self.exact)
                raise NotRepresentableError(
                    "inverse hyperbolic tangent of a non-zero rational is transcendental"
                )
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                if abs(fp.mantissa) >= 10**fp.decimals:  # |x| >= 1 -> outside the open domain
                    raise NotRepresentableError(outside)
                if fp.mantissa == 0:  # atanh(0) = 0, the only exact case
                    return Value(Mode.FIXED_POINT, FixedPoint(0, fp.decimals), exact=self.exact)
                sign = -1 if fp.mantissa < 0 else 1  # atanh is odd; compute on |x|
                magnitude = _fp_atanh(abs(fp.mantissa), fp.decimals)
                return Value(
                    Mode.FIXED_POINT, FixedPoint(sign * magnitude, fp.decimals), exact=False
                )
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    # --- complex (19.1.8) -----------------------------------------------
    # COMPLEX is built ENTIRELY on the FIXED_POINT engine. A complex value is a
    # pair of FixedPoints; every operation below decomposes into fixed-point
    # operations on those parts, wrapped as throwaway FIXED_POINT Values so the
    # existing real add/sub/mul/div/sqrt/exp/sin/cos/... methods can be called and
    # recombined. The covering-scale rule, half-to-even rounding and exactness
    # propagation are therefore inherited per part rather than reinvented, and a
    # landmark like (3+4i)*(1+2i) or sqrt(-1) stays EXACT because each part's real
    # arithmetic does. The transcendentals use the standard real decompositions
    # (exp(a+bi)=e^a(cos b+i sin b), log z=ln|z|+i·arg z, ...).

    def _parts(self) -> tuple["Value", "Value"]:
        """This complex value's (real, imag) parts as FIXED_POINT Values.

        The whole value carries ONE ``exact`` flag, so both parts inherit it; a
        recombining op ANDs the result parts' flags back into one (exact only
        where every part stayed exact), keeping the single-flag contract.
        """
        assert isinstance(self.payload, Complex)
        return (
            Value(Mode.FIXED_POINT, self.payload.real, exact=self.exact),
            Value(Mode.FIXED_POINT, self.payload.imag, exact=self.exact),
        )

    @staticmethod
    def _complex_of(re: "Value", im: "Value") -> "Value":
        """Pack two FIXED_POINT result Values into one COMPLEX Value (exact iff both)."""
        assert isinstance(re.payload, FixedPoint) and isinstance(im.payload, FixedPoint)
        return Value(Mode.COMPLEX, Complex(re.payload, im.payload), exact=re.exact and im.exact)

    @staticmethod
    def _fp_zero(like: "Value") -> "Value":
        """An exact FIXED_POINT zero at ``like``'s scale — the imag part of a real result."""
        assert isinstance(like.payload, FixedPoint)
        return Value(Mode.FIXED_POINT, FixedPoint(0, like.payload.decimals), exact=True)

    @staticmethod
    def _complex_int(n: int) -> "Value":
        """The real integer ``n`` as a scale-0 COMPLEX value (n + 0i), exact."""
        z = Value(Mode.FIXED_POINT, FixedPoint(n, 0), exact=True)
        return Value._complex_of(z, Value._fp_zero(z))

    @staticmethod
    def _fp_nonneg(v: "Value") -> "Value":
        """Clamp a FIXED_POINT radicand to >= 0, healing a tiny negative from rounding.

        sqrt's two radicands (|z|±re)/2 are non-negative in exact arithmetic, but
        the magnitude |z| is itself a rounded root, so it can dip a half-ULP below
        |re| and push a radicand microscopically negative — which real sqrt would
        refuse. Flooring at zero keeps the principal square root defined.
        """
        assert isinstance(v.payload, FixedPoint)
        if v.payload.mantissa < 0:
            return Value(Mode.FIXED_POINT, FixedPoint(0, v.payload.decimals), exact=False)
        return v

    def _complex_add(self, other: "Value") -> "Value":
        (ar, ai), (br, bi) = self._parts(), other._parts()
        return Value._complex_of(ar.add(br), ai.add(bi))

    def _complex_sub(self, other: "Value") -> "Value":
        (ar, ai), (br, bi) = self._parts(), other._parts()
        return Value._complex_of(ar.sub(br), ai.sub(bi))

    def _complex_mul(self, other: "Value") -> "Value":
        (ar, ai), (br, bi) = self._parts(), other._parts()
        # (ar+ai·i)(br+bi·i) = (ar·br - ai·bi) + (ar·bi + ai·br)·i
        return Value._complex_of(ar.mul(br).sub(ai.mul(bi)), ar.mul(bi).add(ai.mul(br)))

    def _complex_div(self, other: "Value") -> "Value":
        (ar, ai), (br, bi) = self._parts(), other._parts()
        denom = br.mul(br).add(bi.mul(bi))  # |other|^2, a real fixed-point value
        assert isinstance(denom.payload, FixedPoint)
        if denom.payload.mantissa == 0:
            raise ZeroDivisionError("complex division by zero")
        # multiply by the conjugate, then scale by 1/|other|^2
        re = ar.mul(br).add(ai.mul(bi)).div(denom)
        im = ai.mul(br).sub(ar.mul(bi)).div(denom)
        return Value._complex_of(re, im)

    def _complex_neg(self) -> "Value":
        re, im = self._parts()
        return Value._complex_of(re.neg(), im.neg())

    def _complex_pos(self) -> "Value":
        re, im = self._parts()
        return Value._complex_of(re.pos(), im.pos())

    def _complex_conj(self) -> "Value":
        re, im = self._parts()
        return Value._complex_of(re, im.neg())

    def _complex_re(self) -> "Value":
        re, _im = self._parts()
        return Value._complex_of(re, Value._fp_zero(re))

    def _complex_im(self) -> "Value":
        _re, im = self._parts()
        return Value._complex_of(im, Value._fp_zero(im))

    def _complex_magnitude(self) -> "Value":
        """|z| as a real FIXED_POINT value — sqrt(re^2 + im^2), the abs/log radius."""
        re, im = self._parts()
        return re.mul(re).add(im.mul(im)).sqrt()

    def _complex_abs(self) -> "Value":
        mag = self._complex_magnitude()
        return Value._complex_of(mag, Value._fp_zero(mag))

    def _complex_arg(self) -> "Value":
        re, im = self._parts()
        ang = im.atan2(re)  # atan2(y, x): self is y, other is x
        return Value._complex_of(ang, Value._fp_zero(ang))

    def _complex_sqrt(self) -> "Value":
        """Principal square root via the algebraic real-part form.

        sqrt(z) = sqrt((|z|+re)/2) + sign(im)·sqrt((|z|-re)/2)·i, with sign(im) taken
        as +1 when im == 0 (the principal branch). Algebraic (not polar) so the grid
        landmarks stay exact: sqrt(-1) = i, sqrt(3+4i) = 2+i.
        """
        re, im = self._parts()
        assert isinstance(im.payload, FixedPoint)
        mag = self._complex_magnitude()
        two = Value(Mode.FIXED_POINT, FixedPoint(2, 0), exact=True)
        re_part = Value._fp_nonneg(mag.add(re).div(two)).sqrt()
        im_mag = Value._fp_nonneg(mag.sub(re).div(two)).sqrt()
        im_part = im_mag if im.payload.mantissa >= 0 else im_mag.neg()
        return Value._complex_of(re_part, im_part)

    def _complex_exp(self) -> "Value":
        re, im = self._parts()
        ex = re.exp()  # e^re, real
        return Value._complex_of(ex.mul(im.cos()), ex.mul(im.sin()))

    def _complex_ln(self) -> "Value":
        """Natural log: ln|z| + arg(z)·i (principal branch). Refuses z == 0."""
        re, im = self._parts()
        lnr = self._complex_magnitude().ln()  # ln raises on |z| == 0
        ang = im.atan2(re)
        return Value._complex_of(lnr, ang)

    def _complex_log(self, base: "Value | None") -> "Value":
        ln_z = self._complex_ln()
        if base is None:
            return ln_z
        return ln_z._complex_div(base._complex_ln())

    def _complex_log_int_base(self, base: int) -> "Value":
        """log base ``base`` (10, 2) as complex ln(z)/ln(base).

        The base is materialised at the operand's own scale, not scale 0 — otherwise
        ln(base) would be computed on a scale-0 integer and round catastrophically
        (ln(10) -> 2), poisoning the quotient.
        """
        assert isinstance(self.payload, Complex)
        scale = max(self.payload.real.decimals, self.payload.imag.decimals)
        base_v = Value(
            Mode.COMPLEX,
            Complex(FixedPoint(base * 10**scale, scale), FixedPoint(0, scale)),
            exact=True,
        )
        return self._complex_ln()._complex_div(base_v._complex_ln())

    def _complex_sin(self) -> "Value":
        # sin(a+bi) = sin a·cosh b + i·cos a·sinh b
        re, im = self._parts()
        return Value._complex_of(re.sin().mul(im.cosh()), re.cos().mul(im.sinh()))

    def _complex_cos(self) -> "Value":
        # cos(a+bi) = cos a·cosh b - i·sin a·sinh b
        re, im = self._parts()
        return Value._complex_of(re.cos().mul(im.cosh()), re.sin().mul(im.sinh()).neg())

    def _complex_tan(self) -> "Value":
        return self._complex_sin()._complex_div(self._complex_cos())

    def _complex_sinh(self) -> "Value":
        # sinh(a+bi) = sinh a·cos b + i·cosh a·sin b
        re, im = self._parts()
        return Value._complex_of(re.sinh().mul(im.cos()), re.cosh().mul(im.sin()))

    def _complex_cosh(self) -> "Value":
        # cosh(a+bi) = cosh a·cos b + i·sinh a·sin b
        re, im = self._parts()
        return Value._complex_of(re.cosh().mul(im.cos()), re.sinh().mul(im.sin()))

    def _complex_tanh(self) -> "Value":
        return self._complex_sinh()._complex_div(self._complex_cosh())

    def _complex_is_real_integer(self) -> int | None:
        """The exponent's integer value when it is a real whole number, else None."""
        assert isinstance(self.payload, Complex)
        re, im = self.payload.real, self.payload.imag
        if im.mantissa != 0:
            return None
        if re.mantissa % 10**re.decimals != 0:
            return None
        return re.mantissa // 10**re.decimals

    def _complex_pow(self, other: "Value") -> "Value":
        """z ** w. An integer real exponent uses exact repeated multiplication
        (squaring); any other exponent goes through exp(w·ln z), inexact."""
        n = other._complex_is_real_integer()
        if n is not None:
            if n == 0:
                return Value._complex_int(1)
            base = self if n > 0 else Value._complex_int(1)._complex_div(self)
            result = Value._complex_int(1)
            factor = base
            k = abs(n)
            while k:  # exponentiation by squaring over complex mul
                if k & 1:
                    result = result._complex_mul(factor)
                k >>= 1
                if k:
                    factor = factor._complex_mul(factor)
            return result
        # general branch: z**w = exp(w · ln z); ln refuses z == 0
        return other._complex_mul(self._complex_ln())._complex_exp()

    # --- reporting (25.1) -----------------------------------------------

    def precision(self) -> int | None:
        """The value's declared decimal precision, when the mode has one (25.1).

        Fixed-point carries an explicit scale — the count of fractional digits
        its mantissa is quantized to — which is exactly the precision a result
        was rounded at; return it (always >= 0, even for an exact result, e.g. 0
        for a whole-number fixed-point). Floating-point and rational have no
        per-value decimal scale (a double's precision is binary; a Fraction is
        exact), so return None — "precision when known" (25.1). Per-mode
        dispatch, one case per type, like every operation above.
        """
        match self.mode:
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                return self.payload.decimals
            case Mode.FLOATING_POINT | Mode.RATIONAL:
                return None
            case Mode.COMPLEX:
                # the wider of the two parts' scales — the precision the result was
                # rounded at, so the inexact verdict reads "rounded to N decimals".
                assert isinstance(self.payload, Complex)
                return max(self.payload.real.decimals, self.payload.imag.decimals)
            case Mode.VECTOR:
                # A vector has no single decimal scale — each element carries its own;
                # the envelope drops the `[scale]` tag (the elements show their own).
                return None
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def to_float(self) -> float:
        """Reduce this Value to a plain ``float`` for the solver's search (31.7).

        The golden-section engine drives the unknown over a real interval and
        compares candidates as floats, so it needs the mode's value collapsed to a
        double. Each mode reads its own payload: floating-point IS a double already;
        fixed-point and rational go through ``Fraction`` so the conversion rounds
        once, correctly, rather than compounding integer-division error. The result
        is a lossy view used ONLY to steer the search — the faithful answer is the
        Value itself, never this float. One case per type, like every operation here.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                return self.payload
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                return float(Fraction(self.payload.mantissa, 10**self.payload.decimals))
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                return float(self.payload)
            case Mode.COMPLEX:
                raise NotRepresentableError("a complex value has no single real float")
            case Mode.VECTOR:
                raise NotRepresentableError("a vector has no single real float")
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    # --- formatting (19.4) ----------------------------------------------

    def to_string(self, scale: int | None = None) -> str:
        """Render in the mode's normal string format (19.4, 10.3).

        Named (not __str__) so it can take formatting options later, per the
        no-overloading rationale; __str__ can delegate here if convenient.

        ``scale``, when given, renders a FIXED_POINT value padded to that many
        fractional digits — the "active precision" the abort diagnostic shows its
        operands at (35.3.1), so an operand written ``3`` reads as ``3.00`` beside a
        scale-2 result. It only ever WIDENS (it takes the max with the value's own
        scale), so the extra digits are zeros an exact value already implies; it is
        ignored outside fixed-point, where there is no decimal scale.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                # str(float) is the shortest round-tripping decimal (10.3).
                return str(self.payload)
            case Mode.FIXED_POINT:
                # Render the mantissa as a plain integer string, then place the
                # decimal point `decimals` digits from the right by pure string
                # manipulation (10.3). Trailing zeros are kept (the declared
                # scale is significant): FixedPoint(150, 2) -> "1.50". A wider
                # requested scale pads the mantissa (exact zero-padding) so the
                # operand shows at the active precision (35.3.1).
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                decimals = fp.decimals if scale is None else max(scale, fp.decimals)
                mantissa = fp.mantissa * 10 ** (decimals - fp.decimals)
                sign = "-" if mantissa < 0 else ""
                digits = str(abs(mantissa))
                if decimals == 0:
                    return sign + digits
                digits = digits.zfill(decimals + 1)  # ensure a leading 0 if needed
                return f"{sign}{digits[:-decimals]}.{digits[-decimals:]}"
            case Mode.RATIONAL:
                # Fraction renders as "n" when integral, else "n/d" (10.3).
                return str(self.payload)
            case Mode.COMPLEX:
                # Render as a+bi: each part via the fixed-point formatter, the imag
                # part carrying an explicit sign and an `i`. A zero imag collapses to
                # the bare real (a real result), a zero real to the bare imaginary
                # (so sqrt(-1) reads "1i", not "0+1i").
                assert isinstance(self.payload, Complex)
                re_fp, im_fp = self.payload.real, self.payload.imag
                re_str = Value(Mode.FIXED_POINT, re_fp, exact=True).to_string()
                if im_fp.mantissa == 0:
                    return re_str
                im_mag = Value(
                    Mode.FIXED_POINT, FixedPoint(abs(im_fp.mantissa), im_fp.decimals), exact=True
                ).to_string()
                if re_fp.mantissa == 0:
                    return f"{'-' if im_fp.mantissa < 0 else ''}{im_mag}i"
                return f"{re_str}{'-' if im_fp.mantissa < 0 else '+'}{im_mag}i"
            case Mode.VECTOR:
                # Bracketed, comma-separated, each element in its own mode's format —
                # the same `[a, b, …]` spelling the literal is written in (`[]` empty).
                assert isinstance(self.payload, Vector)
                return "[" + ", ".join(item.to_string() for item in self.payload.elements) + "]"
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    # --- analyze-tree description (26) -----------------------------------

    def describe(self) -> str:
        """One-line analyze-tree description: value, type[scale], exactness, and
        the mode's hex / exact-decimal / approximation details (26.1, 26.3-26.7).

        The envelope ``<value> (<type>[<scale>], <verdict>)`` is generic — the
        rendered value, the type name, the fixed-point scale when the mode has one
        (26.1), and the exact/inexact verdict (26.5, read straight off this Value's
        own ``exact`` flag). The trailing ` · `-separated fragments are per-mode and
        live in details(). Used by Node.pretty(); not __str__, per 19.4's no-dunder
        rationale (it may grow options later).
        """
        scale = self.precision()
        type_part = self.mode.value + (f"[{scale}]" if scale is not None else "")
        verdict = "exact" if self.exact else "inexact"
        parts = [f"{self.to_string()} ({type_part}, {verdict})", *self.details()]
        if self.error is not None:  # "how inexact": the exact residual this rounding introduced
            # Labelled "rounding", not "error" — the reply's `error` field is the failure
            # channel, so `· error -1/2` would read as "this node failed" (it did not).
            parts.append(f"rounding {self.error} ≈ {_approx_decimal(self.error)}")
        return " · ".join(parts)

    def explain_inexact(self) -> str:
        """A SHORT account of WHY this value is inexact, for the abort message (35.2.2).

        The "why" half of the abort-on-inexact diagnostic — the node walk supplies the
        WHERE (line + sub-expression), this the kind (35.1.1) and magnitude (35.1.2).
        It states only what is CERTAIN at the abort's introduction site (where, by
        construction, every operand was exact), and never guesses:

        - floating-point flags every result inexact, so the only true steer is to a
          type that CAN be exact.
        - fixed-point with a recorded ``error`` rounded an algebraic result (`/`, `*`,
          integer `**`). Its true value is therefore rational, so RATIONAL MODE IS
          EXACT for it — a guaranteed fix we can promise. Whether more fixed-point
          precision would help instead depends on whether the value terminates in
          decimal (10/3 never does), which we do NOT decide here — so we don't claim it.
        - fixed-point with NO error is irrational (a root, trig, or log value): no
          exact representation exists in ANY type, so we promise no fix.

        Only ever called on an inexact value — explaining an exact one is meaningless.
        """
        match self.mode:
            case Mode.FLOATING_POINT:
                return "every floating-point result is inexact; use fixed-point or rational"
            case Mode.FIXED_POINT:
                scale = self.precision()
                unit = "decimal" if scale == 1 else "decimals"
                if self.error is not None:
                    return f"rounded to {scale} {unit}, off by {self.error}; rational mode is exact"
                return "irrational; no exact value in any numeric type"
            case Mode.RATIONAL:
                return "inherited from an inexact input"
            case Mode.COMPLEX:
                return (
                    "a complex part rounded onto the fixed-point grid (a root/transcendental, "
                    "or a quotient that did not fit the scale)"
                )
            case Mode.VECTOR:
                return "a vector element is inexact"
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def hex_dump(self) -> str | None:
        """The value's stored bits as a hex string, or None when there is no single
        integer to dump (26.3, 27.5).

        Fixed-point renders its mantissa in whole-byte hex with M@D notation (the
        same M@D the lexer accepts; ``@<scale>`` dropped at scale 0, so an integer
        fixed-point reads as a plain "0x64"). Floating-point renders the raw 64-bit
        IEEE-754 pattern. Rational is a numerator/denominator pair — no single
        integer to dump — so it returns None; the hex dump is meaningful only for
        the bit-backed types. Shared by details() (the analyze tree) and calculate's
        value_hex_dump field, so both render bits the same way. One case per type.
        """
        match self.mode:
            case Mode.FIXED_POINT:
                assert isinstance(self.payload, FixedPoint)
                fp = self.payload
                sign = "-" if fp.mantissa < 0 else ""
                scale = f"@{fp.decimals}" if fp.decimals else ""  # @0 carries nothing
                return f"{sign}0x{_hex_bytes(abs(fp.mantissa))}{scale}"
            case Mode.FLOATING_POINT:
                assert isinstance(self.payload, float)
                raw = struct.pack(">d", self.payload).hex()  # big-endian raw 64-bit pattern
                return f"0x{raw}"
            case Mode.RATIONAL:
                return None
            case Mode.COMPLEX:
                return None  # two parts, no single integer to dump
            case Mode.VECTOR:
                return None  # many elements, no single integer to dump
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def details(self) -> list[str]:
        """Per-mode hex / exact-decimal / approximation fragments for describe().

        One match case per type, like every operation here. The bit-backed types
        (fixed-point, floating-point) show their bits via hex_dump() (26.3).
        Rational has no single integer to dump, so it shows a decimal approximation
        beside the exact fraction (26.7) instead, omitted for an integer whose
        approximation would merely restate it.
        """
        dump = self.hex_dump()
        if dump is not None:
            return [f"hex {dump}"]
        match self.mode:
            case Mode.RATIONAL:
                assert isinstance(self.payload, Fraction)
                if self.payload.denominator == 1:
                    return []  # an integer is its own decimal — nothing to approximate
                return [f"≈ {self._rational_approx()}"]
            case Mode.COMPLEX:
                return []  # the a+bi rendering already carries both parts
            case Mode.VECTOR:
                return []  # the [a, b, …] rendering already carries every element
            case _:
                raise ValueError(f"unsupported mode: {self.mode!r}")

    def _rational_approx(self) -> str:
        """A bounded-precision decimal approximation of the rational payload (26.7)."""
        assert isinstance(self.payload, Fraction)
        return _approx_decimal(self.payload)
