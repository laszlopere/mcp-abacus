# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Unit tests for the argument-validation error formatter (TODO 43.2)."""

import pydantic
import pytest

from mcp_abacus.errors import format_validation_error

_Model = pydantic.create_model(
    "_Model",
    expression=(str, ...),
    precision=(int, ...),
)


def _error(payload) -> pydantic.ValidationError:
    with pytest.raises(pydantic.ValidationError) as info:
        _Model.model_validate(payload)
    return info.value


def test_missing_field_is_named_as_required():
    msg = format_validation_error("calculate", _error({"precision": 0}))
    assert "argument 'expression' is required but was not provided" in msg
    assert msg.startswith("Invalid arguments for tool 'calculate':")


def test_wrong_type_reports_expected_and_received():
    msg = format_validation_error("calculate", _error({"expression": 123, "precision": 0}))
    assert "argument 'expression' expected a string, but received 123 (int)" in msg


def test_integer_field_phrasing():
    msg = format_validation_error("calculate", _error({"expression": "x", "precision": "abc"}))
    assert "argument 'precision' expected an integer" in msg


def test_multiple_errors_are_joined():
    msg = format_validation_error("calculate", _error({}))
    assert "'expression'" in msg and "'precision'" in msg
    assert "; " in msg


def test_no_pydantic_url_leaks():
    msg = format_validation_error("calculate", _error({}))
    assert "pydantic.dev" not in msg
