# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Shared test fixtures: verbose tracing for ./run-tests.sh --verbose.

Under pytest verbosity >= 1 an autouse fixture instruments the expression
engine's seams — tokenize(), parse(), Node.evaluate(), Value.from_lexeme()
and Value binary arithmetic — so the tests narrate what they do: source
expressions, token streams, parse trees and computed values appear in
readable, indented form. The traces are plain prints, so they are visible
when capture is off (run-tests.sh --verbose passes pytest -v -s).
Non-verbose runs are completely untouched.

A second, independent switch — ``--human-readable`` (run-tests.sh adds -s) —
records solver TOOL calls and prints each as a framed Code / Result block: the
abacus program it was given, then the solution it found (or the no-solution /
error message). It targets the solver the way the verbose trace targets the
evaluator, and takes precedence over -v when both are passed.
"""

import functools
import json
import os

import pytest

import mcp_abacus.expr as expr_package
import mcp_abacus.expr.lexer as lexer_module
import mcp_abacus.expr.parser as parser_module
from mcp_abacus.expr.lexer import EOF, Token
from mcp_abacus.expr.nodes import Node
from mcp_abacus.expr.value import Value


def pytest_addoption(parser):
    """Register ``--human-readable`` (off by default; pair with -s to see output)."""
    parser.addoption(
        "--human-readable",
        action="store_true",
        default=False,
        help="Print each solver call as a framed Code / Result block (run-tests.sh adds -s).",
    )


# A framed block is `<RULE Code>` / program / `<RULE Result>` / result / `<RULE>`,
# all at a fixed width so the rules line up however long the program is.
_RULE_WIDTH = 76


def _rule(label: str = "") -> str:
    """A horizontal rule, optionally labelled (``----- Code -------…``)."""
    if not label:
        return "-" * _RULE_WIDTH
    prefix = f"----------- {label} "
    return prefix + "-" * max(0, _RULE_WIDTH - len(prefix))


def _human_readable_trace(request, monkeypatch):
    """Under --human-readable: print each solver call as a Code / Result block.

    Records solver runs at THREE seams so it works wherever a test drives the
    solver: the over-the-wire tool (ClientSession.call_tool, as test_e2e drives a
    server subprocess), the in-process tool (FastMCP mcp.call_tool, as
    test_solver_e2e drives it), and the engine itself (solver.search, when a test
    calls it directly with a parsed node). `calculate` is framed the same Code /
    Result way for the calculate-driven e2e modules (test_functions_e2e at the
    in-process seam, test_vectors_e2e over the wire), each gated by module name.
    The `curve_fit` tool is framed too: its Code block shows the (x, y) data and its
    Result block the fitted equation(s) with error (test_fit_e2e).
    For the engine seam the source text is recovered by tracing parse() and keying
    on id(node), since search() is handed the AST, not the program string. At teardown
    each run prints the abacus program under a "Code" rule and the outcome under a
    "Result" rule: the found ``solution`` on success, or the message (a "No solution…"
    or a rejected request) on failure. Non-solver work passes straight through, so
    the switch is a no-op for every other test.
    """
    from mcp import ClientSession

    import mcp_abacus.solver as solver_module
    from mcp_abacus.server import mcp
    from mcp_abacus.solver import SolverError

    rows: list[tuple[str, str]] = []  # (program source, result line)
    sources: dict[int, str] = {}  # id(root node) -> the source text it was parsed from
    state = {"muted": False}  # mute the raw evaluate seam while a tool/solver seam owns the row

    def _append(expression: str, result: str) -> None:
        rows.append((expression, result))

    def _record(arguments, payload) -> None:
        expression = (arguments or {}).get("expression", "")
        if payload.get("error") is not None:
            _append(expression, payload["error"])
            return
        # Single unknown -> the scalar `solution`; several -> join the `solutions` list
        # so each found unknown shows in the multivariate Result block.
        solution = payload.get("solution")
        if solution is None and payload.get("solutions"):
            solution = ", ".join(
                f"{entry['variable']} = {entry['solution']}" for entry in payload["solutions"]
            )
        _append(expression, "" if solution is None else str(solution))

    def _record_calculate(arguments, payload) -> None:
        # `calculate` returns a per-line `values` breakdown, not a single solution: the
        # Result block lists one "<expr> = <value>" line per answered line (so a grouped
        # program shows every function's result), or the plain error when it aborted.
        expression = (arguments or {}).get("expression", "")
        if payload.get("error") is not None:
            _append(expression, payload["error"])
            return
        lines = [f"{entry['source']} = {entry['value']}" for entry in (payload.get("values") or [])]
        _append(expression, "\n".join(lines))

    def _record_fit(arguments, payload) -> None:
        # `curve_fit` takes (x, y) data, not an `expression`: the Code block shows the data
        # (and the mode, when set), the Result block lists each fitted form as
        # "<form>: y = <equation>  (error <fit_error>)" — or the plain error when refused.
        args = arguments or {}
        code_lines = [f"x = {args.get('x')}", f"y = {args.get('y')}"]
        if args.get("mode") is not None:
            code_lines.append(f"mode = {args['mode']}")
        code = "\n".join(code_lines)
        if payload.get("error") is not None:
            _append(code, payload["error"])
            return
        lines = [
            f"{fit['form']}: y = {fit['equation']}  (error {fit['fit_error']})"
            for fit in (payload.get("fits") or [])
        ]
        _append(code, "\n".join(lines))

    original_mcp_call = mcp.call_tool

    @functools.wraps(original_mcp_call)
    async def traced_mcp_call(name, arguments=None, *args, **kwargs):
        state["muted"] = True  # the tool owns the row; mute its internal evaluate calls
        try:
            result = await original_mcp_call(name, arguments, *args, **kwargs)
        finally:
            state["muted"] = False
        if name == "solver":
            blocks = result[0] if isinstance(result, tuple) else result
            _record(arguments, json.loads(blocks[0].text))
        elif name == "curve_fit":
            blocks = result[0] if isinstance(result, tuple) else result
            _record_fit(arguments, json.loads(blocks[0].text))
        elif name == "calculate" and request.module.__name__ == "test_functions_e2e":
            # The grouped function tests frame each `calculate` program the same way:
            # gated to that module so other calculate-driven tests stay unaffected.
            blocks = result[0] if isinstance(result, tuple) else result
            _record_calculate(arguments, json.loads(blocks[0].text))
        return result

    original_client_call = ClientSession.call_tool

    @functools.wraps(original_client_call)
    async def traced_client_call(self, name, arguments=None, *args, **kwargs):
        state["muted"] = True  # the tool owns the row; mute its internal evaluate calls
        try:
            result = await original_client_call(self, name, arguments, *args, **kwargs)
        finally:
            state["muted"] = False
        if name == "solver":
            _record(arguments, json.loads(result.content[0].text))
        elif name == "calculate" and request.module.__name__ == "test_vectors_e2e":
            # The vector e2e tests drive `calculate` over the wire (client seam), so frame
            # each as a Code / Result block here — the over-the-wire twin of the mcp-seam
            # gating above for test_functions_e2e. Gated to that module so other client-
            # seam calculate tests (test_e2e) stay unaffected.
            _record_calculate(arguments, json.loads(result.content[0].text))
        return result

    # Engine seam: parse() records each node's source; the engine looks it up and
    # frames its outcome the same way the tool reply does (solution + "(approximate)",
    # or the failure message — re-raised so pytest.raises tests still see it). Both
    # engines are traced: search() (golden-section) and nelder_mead() (multivariate).
    original_parse = parser_module.parse
    original_search = solver_module.search
    original_nelder_mead = solver_module.nelder_mead

    def _result_line(result):
        # One unknown -> just the value (the 1-D output, unchanged); several ->
        # "name = value, …" so each found unknown is shown.
        if len(result.solutions) == 1:
            return f"{result.solutions[0][1].to_string()} (approximate)"
        pairs = ", ".join(f"{name} = {value.to_string()}" for name, value in result.solutions)
        return f"{pairs} (approximate)"

    def _trace_engine(original):
        @functools.wraps(original)
        def traced(node, *args, **kwargs):
            expression = sources.get(id(node), "")
            state["muted"] = True  # mute the evaluate seam: the solver owns this row
            try:
                result = original(node, *args, **kwargs)
                _append(expression, _result_line(result))
                return result
            except SolverError as exc:
                _append(expression, exc.message)
                raise
            except Exception as exc:  # EvalError (unset constant) etc. — still worth showing
                _append(expression, getattr(exc, "message", str(exc)))
                raise
            finally:
                state["muted"] = False

        return traced

    @functools.wraps(original_parse)
    def traced_parse(text):
        node = original_parse(text)
        sources[id(node)] = text
        return node

    traced_search = _trace_engine(original_search)
    traced_nelder_mead = _trace_engine(original_nelder_mead)

    monkeypatch.setattr(mcp, "call_tool", traced_mcp_call)
    monkeypatch.setattr(ClientSession, "call_tool", traced_client_call)
    # Patch parse/search/nelder_mead in every namespace holding the original (the
    # source module, the expr package re-export, and the test module's own
    # from-import), so the call the test makes is the one intercepted. The server keeps
    # its OWN bound engines, so an in-process tool call records once, at the tool seam.
    for module in (parser_module, expr_package, request.module):
        if getattr(module, "parse", None) is original_parse:
            monkeypatch.setattr(module, "parse", traced_parse)
    for module in (solver_module, request.module):
        if getattr(module, "search", None) is original_search:
            monkeypatch.setattr(module, "search", traced_search)
        if getattr(module, "nelder_mead", None) is original_nelder_mead:
            monkeypatch.setattr(module, "nelder_mead", traced_nelder_mead)

    # Calculate seam: Node.evaluate frames the computed value the same way. Only the
    # PARSED ROOT (keyed by traced_parse) records — recursive child evaluates aren't
    # in `sources`, so each calculate test yields exactly one block — and the `muted`
    # guard keeps a tool/solver run's internal evaluations from adding stray rows.
    original_evaluate = Node.evaluate

    @functools.wraps(original_evaluate)
    def traced_evaluate(self, *args, **kwargs):
        record = not state["muted"] and id(self) in sources
        try:
            result = original_evaluate(self, *args, **kwargs)
        except Exception as exc:  # EvalError etc. — frame the message like the engine seam
            if record:
                _append(sources[id(self)], getattr(exc, "message", str(exc)))
            raise
        if record:
            verdict = "exact" if result.exact else "inexact"
            _append(sources[id(self)], f"{result.to_string()} ({verdict})")
        return result

    monkeypatch.setattr(Node, "evaluate", traced_evaluate)

    # Mute the e2e subprocess's stderr request-logging so only the blocks print
    # (the same redirect the verbose e2e trace uses); skipped for in-process modules.
    sink = None
    if getattr(request.module, "stdio_client", None) is not None:
        sink = open(os.devnull, "w")
        original_stdio_client = request.module.stdio_client

        @functools.wraps(original_stdio_client)
        def quiet_stdio_client(server, errlog=None):
            return original_stdio_client(server, errlog=sink)

        monkeypatch.setattr(request.module, "stdio_client", quiet_stdio_client)

    try:
        yield
    finally:
        if sink is not None:
            sink.close()

    if rows:
        print()  # step off pytest's "test-id" progress line
        for program, result in rows:
            print(_rule("Code"))
            for line in program.splitlines() or [""]:
                print(f"  {line}")
            print(_rule("Result"))
            for line in result.splitlines() or [""]:
                print(f"  {line}")
            print(_rule())
            print("\n")  # two blank lines between blocks, separating the tests


# Value has no operator dunders (19.5); each binary op is a named method. Map
# method -> source symbol so the tracer can narrate "a op b" arithmetic.
_BINARY_METHOD_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
}


class _Tracer:
    """Indentation-aware printer for the instrumented calls of one test."""

    def __init__(self) -> None:
        self.depth = 0
        self.in_evaluate = False
        self._started = False

    def emit(self, text: str) -> None:
        if not self._started:
            print()  # step off pytest's "test-id" progress line
            self._started = True
        indent = "  " * self.depth
        for line in text.splitlines():
            print(f"{indent}{line}")

    def raised(self, exc: BaseException) -> None:
        if getattr(exc, "_abacus_traced", False):
            return  # an inner instrumented call already reported it
        line = getattr(exc, "line", None)
        where = f" (line {line})" if line is not None else ""
        self.emit(f"raises {type(exc).__name__}: {exc}{where}")
        exc._abacus_traced = True  # type: ignore[attr-defined]


def _render_tokens(tokens: list[Token]) -> str:
    annotate = any(token.line != 1 for token in tokens)
    parts = []
    for token in tokens:
        text = token.kind if token.kind == EOF else f"{token.kind} {token.lexeme!r}"
        if annotate:
            text += f" @line {token.line}"
        parts.append(text)
    return "tokens: " + ", ".join(parts)


def _render_value(value: Value) -> str:
    return f"{value.payload} ({value.mode.value}, {'exact' if value.exact else 'inexact'})"


def _compact_parser_trace(request, monkeypatch):
    """For test_parser.py under -v: print only "expression = value", aligned.

    Suppresses the token-stream and parse-tree narration of the full tracer;
    instead it buffers every top-level parse()+evaluate() pair and, at
    teardown, prints them side-by-side with the '=' columns lined up. A parse
    that is never evaluated (structural / parse-error tests) yields no row, so
    only expressions that produce a value appear. Other modules are untouched.
    """
    rows: list[tuple[str, str]] = []
    sources: dict[int, str] = {}  # id(root node) -> the source text it was parsed from
    in_evaluate = False

    original_parse = parser_module.parse
    original_evaluate = Node.evaluate

    @functools.wraps(original_parse)
    def traced_parse(text):
        node = original_parse(text)  # a ParseError propagates -> no row, as intended
        sources[id(node)] = text
        return node

    @functools.wraps(original_evaluate)
    def traced_evaluate(self, mode, *args, **kwargs):
        nonlocal in_evaluate
        if in_evaluate:
            # only the outermost call records a row
            return original_evaluate(self, mode, *args, **kwargs)
        in_evaluate = True
        expr = sources.get(id(self), repr(self))
        try:
            result = original_evaluate(self, mode, *args, **kwargs)
            rows.append((expr, _render_value(result)))
            return result
        except Exception as exc:
            line = getattr(exc, "line", None)
            where = f" (line {line})" if line is not None else ""
            rows.append((expr, f"raises {type(exc).__name__}: {exc}{where}"))
            raise
        finally:
            in_evaluate = False

    monkeypatch.setattr(Node, "evaluate", traced_evaluate)
    # Patch every namespace holding `parse` — including the test module's own
    # from-import — so the call the test makes routes through traced_parse.
    for module in (parser_module, expr_package, request.module):
        if getattr(module, "parse", None) is original_parse:
            monkeypatch.setattr(module, "parse", traced_parse)

    yield

    if rows:
        width = max(len(expr) for expr, _ in rows)
        print()  # step off pytest's "test-id" progress line
        for expr, rendered in rows:
            print(f"{expr.ljust(width)}  = {rendered}")


def _compact_e2e_trace(request, monkeypatch):
    """For test_e2e.py under -v: print each call as a REQUEST / REPLY pair.

    The e2e tests evaluate inside a server subprocess, out of reach of the in-
    process tracer that the other modules use. So instead we record each
    `calculate` invocation at the client seam (ClientSession.call_tool) and, at
    teardown, print the wire round-trip the way a client sees it: the request
    arguments and the reply payload, each as a 2-space-indented JSON block under
    a REQUEST: / REPLY: heading. Both sides are real JSON, so the block reads the
    same as what travels over stdio. The server's own request logging on stderr
    is muted so only these blocks appear.
    """
    from mcp import ClientSession

    rows: list[tuple[dict, str]] = []
    original_call_tool = ClientSession.call_tool

    @functools.wraps(original_call_tool)
    async def traced_call_tool(self, name, arguments=None, *args, **kwargs):
        result = await original_call_tool(self, name, arguments, *args, **kwargs)
        if name == "calculate":
            rows.append((arguments or {}, result.content[0].text))
        return result

    monkeypatch.setattr(ClientSession, "call_tool", traced_call_tool)

    # Silence the server subprocess's stderr request-logging so nothing but the
    # table prints. test_e2e binds stdio_client by name, so patch the module's
    # reference to redirect the child's stderr into a sink.
    sink = open(os.devnull, "w")
    original_stdio_client = request.module.stdio_client

    @functools.wraps(original_stdio_client)
    def quiet_stdio_client(server, errlog=None):
        return original_stdio_client(server, errlog=sink)

    monkeypatch.setattr(request.module, "stdio_client", quiet_stdio_client)

    try:
        yield
    finally:
        sink.close()

    if rows:

        def _indent(text: str) -> str:
            return "\n".join("  " + line for line in text.splitlines())

        print()  # step off pytest's "test-id" progress line
        for arguments, result_text in rows:
            request_json = json.dumps(arguments, indent=2)
            try:
                reply_json = json.dumps(json.loads(result_text), indent=2)
            except json.JSONDecodeError:
                reply_json = result_text  # non-JSON reply: show it verbatim
            print("\nREQUEST:")
            print(_indent(request_json))
            print("\nREPLY:")
            print(_indent(reply_json))


def _compact_solver_e2e_trace(request, monkeypatch):
    """For test_solver_e2e.py under -v: print each solver tool call as REQUEST / REPLY.

    The tests drive the in-process `solver` tool (mcp.call_tool), so each call is
    recorded at that seam and, at teardown, printed as the request arguments and the
    full reply payload — the actual JSON the tool sends back — each as a 2-space-
    indented block, a blank line separating the cases. Non-solver work passes
    straight through, so other tools in the module record nothing.
    """
    from mcp_abacus.server import mcp

    rows: list[tuple[dict, dict]] = []  # (request args, reply payload)
    original_call_tool = mcp.call_tool

    @functools.wraps(original_call_tool)
    async def traced_call_tool(name, arguments=None, *args, **kwargs):
        result = await original_call_tool(name, arguments, *args, **kwargs)
        if name == "solver":
            blocks = result[0] if isinstance(result, tuple) else result
            rows.append((arguments or {}, json.loads(blocks[0].text)))
        return result

    monkeypatch.setattr(mcp, "call_tool", traced_call_tool)
    yield

    if rows:

        def _indent(text: str) -> str:
            return "\n".join("  " + line for line in text.splitlines())

        print()  # step off pytest's "test-id" progress line
        for request_args, reply in rows:
            print("\nREQUEST:")
            print(_indent(json.dumps(request_args, indent=2)))
            print("\nREPLY:")
            print(_indent(json.dumps(reply, indent=2)))
            print("\n")  # blank line(s) between cases


def _compact_functions_e2e_trace(request, monkeypatch):
    """For test_functions_e2e.py under -v: print each calculate call as REQUEST / REPLY.

    The grouped tests drive the in-process `calculate` tool (mcp.call_tool) with multi-
    line programs, so each call is recorded at that seam and, at teardown, printed as
    the request arguments and the full reply payload — the actual JSON the tool sends
    back, `values` array and all — each as a 2-space-indented block, a blank line
    separating the cases. Mirrors _compact_solver_e2e_trace, for `calculate`.
    """
    from mcp_abacus.server import mcp

    rows: list[tuple[dict, dict]] = []  # (request args, reply payload)
    original_call_tool = mcp.call_tool

    @functools.wraps(original_call_tool)
    async def traced_call_tool(name, arguments=None, *args, **kwargs):
        result = await original_call_tool(name, arguments, *args, **kwargs)
        if name == "calculate":
            blocks = result[0] if isinstance(result, tuple) else result
            rows.append((arguments or {}, json.loads(blocks[0].text)))
        return result

    monkeypatch.setattr(mcp, "call_tool", traced_call_tool)
    yield

    if rows:

        def _indent(text: str) -> str:
            return "\n".join("  " + line for line in text.splitlines())

        print()  # step off pytest's "test-id" progress line
        for request_args, reply in rows:
            print("\nREQUEST:")
            print(_indent(json.dumps(request_args, indent=2)))
            print("\nREPLY:")
            print(_indent(json.dumps(reply, indent=2)))
            print("\n")  # blank line(s) between cases


def _compact_fit_e2e_trace(request, monkeypatch):
    """For test_fit_e2e.py under -v: print each fit call as REQUEST / REPLY.

    The tests drive the in-process `curve_fit` tool (mcp.call_tool), so each call is
    recorded at that seam and, at teardown, printed as the request arguments and the full
    reply payload — the actual JSON the tool sends back, `fits` array and all — each as a
    2-space-indented block, a blank line separating the cases. Mirrors
    _compact_solver_e2e_trace / _compact_functions_e2e_trace, for `curve_fit`.
    """
    from mcp_abacus.server import mcp

    rows: list[tuple[dict, dict]] = []  # (request args, reply payload)
    original_call_tool = mcp.call_tool

    @functools.wraps(original_call_tool)
    async def traced_call_tool(name, arguments=None, *args, **kwargs):
        result = await original_call_tool(name, arguments, *args, **kwargs)
        if name == "curve_fit":
            blocks = result[0] if isinstance(result, tuple) else result
            rows.append((arguments or {}, json.loads(blocks[0].text)))
        return result

    monkeypatch.setattr(mcp, "call_tool", traced_call_tool)
    yield

    if rows:

        def _indent(text: str) -> str:
            return "\n".join("  " + line for line in text.splitlines())

        print()  # step off pytest's "test-id" progress line
        for request_args, reply in rows:
            print("\nREQUEST:")
            print(_indent(json.dumps(request_args, indent=2)))
            print("\nREPLY:")
            print(_indent(json.dumps(reply, indent=2)))
            print("\n")  # blank line(s) between cases


def _compact_functions_trace(request, monkeypatch):
    """For test_functions.py under -v: print each call as "expression [mode] = value".

    Most function tests drive the in-process `calculate` tool, so we record each
    invocation at the FastMCP seam (mcp.call_tool) and print the tool's annotated
    result value (e.g. "16 (exact)") beside its expression — or the line-tagged
    message when the call refused. A few tests (the time() epoch-injection ones,
    28.1.2) cannot go through the tool because they pass a fixed now_ns, so they
    call parse()+evaluate() directly; those are recorded at the evaluate seam
    instead, showing the rendered value the same compact way. The in_tool guard
    keeps a tool call's internal evaluate() from also recording a duplicate row.
    Modes are shown only when set (the tool) or always (direct evaluate).
    """
    from mcp_abacus.server import mcp

    rows: list[tuple[str, str]] = []
    sources: dict[int, str] = {}  # id(root node) -> the source text it was parsed from
    in_tool = False
    in_evaluate = False

    original_call_tool = mcp.call_tool
    original_parse = parser_module.parse
    original_evaluate = Node.evaluate

    @functools.wraps(original_call_tool)
    async def traced_call_tool(name, arguments=None, *args, **kwargs):
        nonlocal in_tool
        in_tool = True
        try:
            result = await original_call_tool(name, arguments, *args, **kwargs)
        finally:
            in_tool = False
        if name == "calculate":
            blocks = result[0] if isinstance(result, tuple) else result
            payload = json.loads(blocks[0].text)
            arguments = arguments or {}
            mode = arguments.get("mode")
            label = arguments["expression"] + (f"  [{mode}]" if mode else "")
            shown = payload["value"] if payload["error"] is None else f"error: {payload['error']}"
            rows.append((label, shown))
        return result

    @functools.wraps(original_parse)
    def traced_parse(text):
        node = original_parse(text)
        sources[id(node)] = text
        return node

    @functools.wraps(original_evaluate)
    def traced_evaluate(self, mode, *args, **kwargs):
        nonlocal in_evaluate
        if in_tool or in_evaluate:
            # a tool call records at its own seam; nested evaluate()s don't recurse a row
            return original_evaluate(self, mode, *args, **kwargs)
        in_evaluate = True
        label = f"{sources.get(id(self), repr(self))}  [{mode.value}]"
        try:
            result = original_evaluate(self, mode, *args, **kwargs)
            verdict = "exact" if result.exact else "inexact"
            rows.append((label, f"{result.to_string()} ({verdict})"))
            return result
        except Exception as exc:
            line = getattr(exc, "line", None)
            where = f" (line {line})" if line is not None else ""
            rows.append((label, f"error: {type(exc).__name__}: {exc}{where}"))
            raise
        finally:
            in_evaluate = False

    monkeypatch.setattr(mcp, "call_tool", traced_call_tool)
    monkeypatch.setattr(Node, "evaluate", traced_evaluate)
    for module in (parser_module, expr_package, request.module):
        if getattr(module, "parse", None) is original_parse:
            monkeypatch.setattr(module, "parse", traced_parse)
    yield

    if rows:
        width = max(len(label) for label, _ in rows)
        print()  # step off pytest's "test-id" progress line
        for label, shown in rows:
            print(f"{label.ljust(width)}  = {shown}")


def _compact_variables_trace(request, monkeypatch):
    """For test_variables_e2e.py under -v: print each multi-line program + result.

    Variable programs span several source lines, so the single-line "expression =
    value" table the other tool tests use would mangle them. Instead each
    `calculate` invocation is recorded at the FastMCP seam (mcp.call_tool) and, at
    teardown, printed as an INPUT block (the program's own lines, indented) under
    its mode, followed by the OUTPUT line — the annotated value, or the line-tagged
    message when the program refused.
    """
    from mcp_abacus.server import mcp

    rows: list[tuple[str, str, str]] = []  # (program source, mode label, shown result)
    original_call_tool = mcp.call_tool

    @functools.wraps(original_call_tool)
    async def traced_call_tool(name, arguments=None, *args, **kwargs):
        result = await original_call_tool(name, arguments, *args, **kwargs)
        if name == "calculate":
            blocks = result[0] if isinstance(result, tuple) else result
            payload = json.loads(blocks[0].text)
            arguments = arguments or {}
            mode = arguments.get("mode", "fixed-point (default)")
            shown = payload["value"] if payload["error"] is None else f"error: {payload['error']}"
            rows.append((arguments["expression"], mode, shown))
        return result

    monkeypatch.setattr(mcp, "call_tool", traced_call_tool)
    yield

    if rows:
        print()  # step off pytest's "test-id" progress line
        for program, mode, shown in rows:
            print(f"\nINPUT [{mode}]:")
            for line in program.splitlines():
                print(f"    {line}")
            print(f"OUTPUT: {shown}")


@pytest.fixture(autouse=True)
def _verbose_trace(request, monkeypatch):
    """Under pytest -v, narrate expressions, tokens, trees and values live."""
    if request.config.option.human_readable:
        # An independent switch (and -v's superior for solver runs): frame each
        # solver call as a Code / Result block instead of the engine trace.
        yield from _human_readable_trace(request, monkeypatch)
        return

    if request.config.option.verbose < 1:
        yield
        return

    if request.module.__name__ == "test_parser":
        # test_parser.py gets the compact "expression = value" side-by-side view.
        yield from _compact_parser_trace(request, monkeypatch)
        return

    if request.module.__name__ == "test_e2e":
        # test_e2e.py gets the compact "expression = result" view, sourced from
        # the client seam since evaluation happens in a subprocess.
        yield from _compact_e2e_trace(request, monkeypatch)
        return

    if request.module.__name__ == "test_solver_e2e":
        # test_solver_e2e.py gets the REQUEST / REPLY JSON view, recorded at the in-
        # process `solver` tool seam so the pair is the real request and the real reply
        # the tool sends back. (test_solver.py is unit-level and drives no tool.)
        yield from _compact_solver_e2e_trace(request, monkeypatch)
        return

    if request.module.__name__ == "test_functions":
        # test_functions.py gets the compact "expression [mode] = value" view,
        # sourced from the in-process `calculate` tool seam.
        yield from _compact_functions_trace(request, monkeypatch)
        return

    if request.module.__name__ == "test_functions_e2e":
        # test_functions_e2e.py gets the REQUEST / REPLY JSON view, recorded at the in-
        # process `calculate` tool seam so the pair is the real request and the real
        # multi-line reply (the `values` breakdown) the tool sends back, like solver_e2e.
        yield from _compact_functions_e2e_trace(request, monkeypatch)
        return

    if request.module.__name__ == "test_fit_e2e":
        # test_fit_e2e.py gets the REQUEST / REPLY JSON view, recorded at the in-process
        # `curve_fit` tool seam so the pair is the real request and the real reply (the `fits`
        # breakdown) the tool sends back, like solver_e2e / functions_e2e.
        yield from _compact_fit_e2e_trace(request, monkeypatch)
        return

    if request.module.__name__ == "test_variables_e2e":
        # test_variables_e2e.py gets a multi-line "program -> result" view: the
        # expressions span several lines, so a side-by-side table won't do.
        yield from _compact_variables_trace(request, monkeypatch)
        return

    tracer = _Tracer()

    original_tokenize = lexer_module.tokenize
    original_parse = parser_module.parse
    original_evaluate = Node.evaluate
    original_from_lexeme = Value.from_lexeme

    @functools.wraps(original_tokenize)
    def traced_tokenize(text):
        tracer.emit(f"tokenize({text!r})")
        tracer.depth += 1
        try:
            tokens = original_tokenize(text)
            tracer.emit(_render_tokens(tokens))
            return tokens
        except Exception as exc:
            tracer.raised(exc)
            raise
        finally:
            tracer.depth -= 1

    @functools.wraps(original_parse)
    def traced_parse(text):
        tracer.emit(f"parse({text!r})")
        tracer.depth += 1
        try:
            node = original_parse(text)
            tracer.emit("tree:")
            tracer.depth += 1
            tracer.emit(node.pretty())
            tracer.depth -= 1
            return node
        except Exception as exc:
            tracer.raised(exc)
            raise
        finally:
            tracer.depth -= 1

    @functools.wraps(original_evaluate)
    def traced_evaluate(self, mode, *args, **kwargs):
        if tracer.in_evaluate:
            # only the outermost call narrates
            return original_evaluate(self, mode, *args, **kwargs)
        tracer.emit(f"evaluate(mode={mode!r})")
        tracer.in_evaluate = True
        tracer.depth += 1
        try:
            result = original_evaluate(self, mode, *args, **kwargs)
            tracer.emit(self.pretty())  # per-node "= value" annotations (18.6)
            tracer.emit(f"value: {_render_value(result)}")
            return result
        except Exception as exc:
            tracer.raised(exc)
            raise
        finally:
            tracer.in_evaluate = False
            tracer.depth -= 1

    @functools.wraps(original_from_lexeme)
    def traced_from_lexeme(lexeme, mode, min_decimals=0):
        if tracer.in_evaluate:  # the evaluate() trace already shows the whole tree
            return original_from_lexeme(lexeme, mode, min_decimals)
        try:
            value = original_from_lexeme(lexeme, mode, min_decimals)
            tracer.emit(f"Value.from_lexeme({lexeme!r}, {mode!r}) -> {_render_value(value)}")
            return value
        except Exception as exc:
            tracer.emit(f"Value.from_lexeme({lexeme!r}, {mode!r})")
            tracer.raised(exc)
            raise

    def _make_traced_binary(op, original):
        @functools.wraps(original)
        def traced_binary(self, other):
            if tracer.in_evaluate:
                return original(self, other)
            other_text = str(other.payload) if isinstance(other, Value) else repr(other)
            try:
                result = original(self, other)
                tracer.emit(f"{self.payload} {op} {other_text} -> {_render_value(result)}")
                return result
            except Exception as exc:
                tracer.emit(f"{self.payload} {op} {other_text}")
                tracer.raised(exc)
                raise

        return traced_binary

    monkeypatch.setattr(Node, "evaluate", traced_evaluate)
    monkeypatch.setattr(Value, "from_lexeme", traced_from_lexeme)
    for _method, _op in _BINARY_METHOD_OPS.items():
        monkeypatch.setattr(Value, _method, _make_traced_binary(_op, getattr(Value, _method)))
    # Patch every namespace holding a reference — including the test module's
    # own globals (from-imports) — with the SAME wrapper objects, so identity
    # checks like `expr.parse is parser.parse` keep holding.
    for module in (lexer_module, parser_module, expr_package, request.module):
        if getattr(module, "tokenize", None) is original_tokenize:
            monkeypatch.setattr(module, "tokenize", traced_tokenize)
        if getattr(module, "parse", None) is original_parse:
            monkeypatch.setattr(module, "parse", traced_parse)
    yield
