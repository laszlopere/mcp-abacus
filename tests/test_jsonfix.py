# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the tolerant argument-repair fallback (TODO 43.1 / 43.2).

Two layers: pure-function unit tests of `repair_arguments` / `process_incoming`,
and raw-protocol end-to-end tests over a real subprocess that prove a
`tools/call` whose `arguments` arrive as a malformed JSON STRING is either
repaired and run (43.1, the FastMCP #932 offender) or, when unparseable,
answered with an actionable JSON-RPC parse error (43.2).
"""

import json
import select
import subprocess
import sys

import pytest
from mcp.shared.message import SessionMessage
from mcp.types import LATEST_PROTOCOL_VERSION, PARSE_ERROR, JSONRPCMessage, JSONRPCRequest

from mcp_abacus.jsonfix import process_incoming, repair_arguments

# --- repair_arguments: the pure core -----------------------------------------


def test_passes_through_a_real_dict_unchanged():
    args = {"expression": "1+2", "mode": "fixed-point"}
    assert repair_arguments(args) is args


def test_parses_a_well_formed_json_string_blob():
    assert repair_arguments('{"expression": "1+2"}') == {"expression": "1+2"}


def test_repairs_single_quotes_and_trailing_comma():
    assert repair_arguments("{'expression': '1+2',}") == {"expression": "1+2"}


def test_repairs_unquoted_bareword_value():
    # The classic offender from TODO 43: "variable": n  ->  "n"
    assert repair_arguments('{"variable": n}') == {"variable": "n"}


def test_unrepairable_garbage_is_returned_unchanged():
    # json-repair coerces junk to "" -- never surface that; hand the original
    # string back so the SDK raises its own informative validation error.
    assert repair_arguments("not json at all") == "not json at all"


def test_a_json_string_that_is_not_an_object_is_left_alone():
    assert repair_arguments('"just a string"') == '"just a string"'
    assert repair_arguments("[1, 2, 3]") == "[1, 2, 3]"


def test_non_string_non_dict_inputs_pass_through():
    assert repair_arguments(None) is None
    assert repair_arguments(42) == 42


# --- process_incoming: the message-level router ------------------------------


def _call_message(arguments) -> SessionMessage:
    req = JSONRPCRequest(
        jsonrpc="2.0",
        id=1,
        method="tools/call",
        params={"name": "calculate", "arguments": arguments},
    )
    return SessionMessage(message=JSONRPCMessage(req))


def test_message_with_string_arguments_is_repaired_and_forwarded():
    out = process_incoming(_call_message("{'expression': '1+2',}"))
    assert out.reply is None
    assert out.forward.message.root.params["arguments"] == {"expression": "1+2"}


def test_message_with_dict_arguments_is_forwarded_untouched():
    msg = _call_message({"expression": "1+2"})
    out = process_incoming(msg)
    assert out.reply is None
    assert out.forward is msg


def test_non_tools_call_message_is_forwarded_untouched():
    req = JSONRPCRequest(jsonrpc="2.0", id=1, method="tools/list", params={})
    msg = SessionMessage(message=JSONRPCMessage(req))
    out = process_incoming(msg)
    assert out.reply is None
    assert out.forward is msg


def test_unparseable_string_arguments_yield_a_parse_error_reply():
    out = process_incoming(_call_message("{bad json"))
    assert out.forward is None
    error = out.reply.message.root.error
    assert error.code == PARSE_ERROR
    assert error.message.lower().count("json") >= 1
    assert "object" in error.message  # tells the model to send an object


# --- raw-protocol end-to-end: the honest proof -------------------------------


class _RawStdioServer:
    """A subprocess speaking newline-delimited JSON-RPC over stdio."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_abacus"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout: float = 10.0) -> dict:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("no JSON-RPC line within timeout")
        return json.loads(self.proc.stdout.readline())

    def recv_id(self, want_id: int, timeout: float = 10.0) -> dict:
        while True:
            msg = self.recv(timeout)
            if msg.get("id") == want_id:
                return msg

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=5)


@pytest.fixture
def raw_server():
    server = _RawStdioServer()
    server.send(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
    )
    server.recv_id(1)
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    try:
        yield server
    finally:
        server.close()


def _call(raw_server, call_id, name, arguments):
    raw_server.send(
        {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return raw_server.recv_id(call_id)


def test_stringified_arguments_run_end_to_end(raw_server):
    # The whole arguments object double-encoded as a string, with single quotes
    # and a trailing comma -- rejected outright without the 43.1 interposer.
    msg = _call(raw_server, 2, "calculate", "{'expression': '6*7', 'mode': 'fixed-point',}")
    result = msg["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["value"].startswith("42")


def test_unparseable_arguments_get_actionable_parse_error(raw_server):
    # 43.2: a string that cannot be parsed into an object -> actionable -32700,
    # not the SDK's bare "Invalid request parameters".
    msg = _call(raw_server, 2, "calculate", "{bad json")
    assert "result" not in msg
    error = msg["error"]
    assert error["code"] == PARSE_ERROR
    assert "JSON object" in error["message"]


def test_missing_required_argument_names_the_field(raw_server):
    # 43.2: reshaped pydantic error -> names the field, no errors.pydantic.dev URL.
    msg = _call(raw_server, 2, "calculate", {"mode": "fixed-point"})
    result = msg["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "'expression' is required" in text
    assert "pydantic.dev" not in text


def test_wrong_argument_type_reports_expected_and_received(raw_server):
    # 43.2: expected-vs-received phrasing.
    msg = _call(raw_server, 2, "calculate", {"expression": 123})
    result = msg["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "'expression' expected a string" in text
    assert "received 123" in text
    assert "pydantic.dev" not in text
