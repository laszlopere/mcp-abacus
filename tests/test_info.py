# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Tests for the `info` tool's return payload (called as a plain function)."""

from importlib.metadata import version

from mcp_abacus.server import info


def test_returns_dict_with_available_status():
    payload = info()
    assert isinstance(payload, dict)
    assert payload["status"] == "available"


def test_name():
    assert info()["name"] == "mcp-abacus"


def test_version_matches_installed_metadata():
    v = info()["version"]
    assert isinstance(v, str)
    assert v
    assert v == version("mcp-abacus")


def test_python_and_mcp_sdk_present_and_nonempty():
    payload = info()
    assert isinstance(payload["python"], str)
    assert payload["python"]
    assert isinstance(payload["mcp_sdk"], str)
    assert payload["mcp_sdk"]


def test_toolsets_is_a_list_empty_for_now():
    payload = info()
    assert isinstance(payload["toolsets"], list)
    assert payload["toolsets"] == []
