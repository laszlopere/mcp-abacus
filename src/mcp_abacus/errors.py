# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Reshape SDK argument-validation errors into actionable text (TODO 43.2).

When a tool call's arguments parse as JSON but fail the tool's schema (a missing
field, a wrong type), FastMCP raises a pydantic ValidationError whose default
string carries a multi-line preamble and a `errors.pydantic.dev` URL -- noise an
LLM cannot act on. `format_validation_error` rebuilds the message from the
structured `.errors()` list into one line that names each offending argument and
what it expected versus what it received, so the model self-corrects in one turn
instead of looping. Server wiring (the FastMCP subclass that catches the error)
lives in server.py.
"""

from __future__ import annotations

from pydantic import ValidationError

# pydantic v2 error `type` -> the human phrasing of what the field expected.
# Anything not listed falls back to pydantic's own short `msg` (URL-free).
_EXPECTED = {
    "string_type": "a string",
    "int_type": "an integer",
    "int_parsing": "an integer",
    "int_from_float": "an integer",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "a boolean",
    "bool_parsing": "a boolean",
    "list_type": "an array",
    "dict_type": "an object",
}


def format_validation_error(tool_name: str, exc: ValidationError) -> str:
    """One-line, self-correcting summary of an argument ValidationError."""
    parts = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = ".".join(str(p) for p in loc) if loc else "arguments"
        etype = err.get("type", "")
        if etype == "missing":
            parts.append(f"argument {field!r} is required but was not provided")
            continue
        received = err.get("input")
        expected = _EXPECTED.get(etype)
        if expected is not None:
            parts.append(
                f"argument {field!r} expected {expected}, but received "
                f"{received!r} ({type(received).__name__})"
            )
        else:
            detail = err.get("msg", "invalid value")
            parts.append(f"argument {field!r}: {detail} (received {received!r})")
    return f"Invalid arguments for tool {tool_name!r}: {'; '.join(parts)}."
