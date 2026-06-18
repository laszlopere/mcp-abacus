# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tolerant repair of LLM-mangled tool-call arguments (TODO 43.1).

LLMs sometimes send a tool call's `arguments` as a JSON *string* instead of an
object -- often the whole blob double-encoded -- and that string may use single
quotes, trailing commas, or unquoted barewords. The MCP SDK validates the typed
`CallToolRequest` strictly, so such input is rejected before any tool runs.

This module repairs the loosely-typed `JSONRPCRequest.params["arguments"]` of an
incoming message BEFORE the session's strict validation sees it -- the one place
the offending bytes are still a plain string. `process_incoming` is wired into
the stdio read stream by the entry point (`__main__`); `repair_arguments` is the
pure, mode-agnostic core. Well-formed input is forwarded untouched.

When the arguments are a string that cannot be parsed into an object at all
(TODO 43.2), `process_incoming` answers the request directly with an actionable
JSON-RPC parse error instead of letting the SDK return a bare "Invalid request
parameters" -- so the model is told to send a JSON object and self-corrects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from json_repair import loads as _repair_loads
from mcp.shared.message import SessionMessage
from mcp.types import PARSE_ERROR, ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCRequest


def repair_arguments(arguments: Any) -> Any:
    """Coerce a stringified / malformed `arguments` blob into a dict.

    When `arguments` is a string, try strict json.loads first, then json-repair
    (which fixes single quotes, trailing commas, and unquoted barewords). Return
    the parsed dict on success. Anything already a dict -- or that does not
    repair to a dict -- is returned UNCHANGED, so genuinely broken input still
    reaches the SDK and raises its normal, informative validation error rather
    than being silently turned into junk.
    """
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed: Any = json.loads(arguments)
    except ValueError:
        try:
            parsed = _repair_loads(arguments)
        except ValueError:
            return arguments
    return parsed if isinstance(parsed, dict) else arguments


@dataclass(frozen=True)
class Incoming:
    """How the stdio interposer should handle one inbound message.

    Exactly one field is set. `forward` is the (possibly repaired) message to
    hand on to the session; `reply` is an actionable error to send straight back
    to the client when the arguments could not be parsed into an object at all.
    """

    forward: SessionMessage | None = None
    reply: SessionMessage | None = None


def process_incoming(message: SessionMessage) -> Incoming:
    """Repair, forward, or reject one inbound `tools/call` message.

    A `tools/call` whose `arguments` arrive as a STRING is repaired into an
    object when possible (43.1) and forwarded; if it cannot be parsed into an
    object, the request is answered directly with an actionable JSON-RPC parse
    error (43.2). Every other message -- well-formed calls, calls whose
    `arguments` are already an object, and non-`tools/call` traffic -- is
    forwarded untouched.
    """
    root = message.message.root
    if not isinstance(root, JSONRPCRequest) or root.method != "tools/call":
        return Incoming(forward=message)
    params = root.params
    if not isinstance(params, dict) or not isinstance(params.get("arguments"), str):
        return Incoming(forward=message)

    repaired = repair_arguments(params["arguments"])
    if isinstance(repaired, dict):
        new_root = root.model_copy(update={"params": {**params, "arguments": repaired}})
        forwarded = SessionMessage(message=JSONRPCMessage(new_root), metadata=message.metadata)
        return Incoming(forward=forwarded)

    return Incoming(reply=_parse_error_reply(root.id, params.get("name")))


def _parse_error_reply(request_id: Any, tool_name: Any) -> SessionMessage:
    """An actionable JSON-RPC parse error for unparseable tool `arguments` (43.2)."""
    label = f"tool {tool_name!r}" if isinstance(tool_name, str) else "the tool"
    message = (
        f"The 'arguments' for {label} were not valid JSON. Send `arguments` as a JSON "
        'object -- e.g. {"expression": "1+2"} -- not a quoted string, and check for '
        "unbalanced braces, single quotes, or missing quotes around keys or values."
    )
    error = JSONRPCError(
        jsonrpc="2.0",
        id=request_id,
        error=ErrorData(code=PARSE_ERROR, message=message),
    )
    return SessionMessage(message=JSONRPCMessage(error))
