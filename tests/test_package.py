# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Import / packaging smoke tests."""

import importlib
import importlib.metadata


def test_import_and_version():
    import mcp_abacus

    assert mcp_abacus.__version__
    assert isinstance(mcp_abacus.__version__, str)


def test_unbuilt_version_fallback_does_not_raise(monkeypatch):
    import mcp_abacus

    def raise_not_found(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    try:
        reloaded = importlib.reload(mcp_abacus)
        assert reloaded.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(mcp_abacus)
