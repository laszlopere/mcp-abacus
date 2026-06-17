# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Import / packaging smoke tests."""

import importlib
import importlib.metadata


def test_import_and_version():
    import mcp_abacus

    assert mcp_abacus.__version__
    assert isinstance(mcp_abacus.__version__, str)


def test_package_summary_names_the_three_real_modes_only():
    # TODO 41.4: the distribution Summary (from pyproject `description`) is the
    # PyPI / Glama card, so it must match the engine — name the three modes that
    # actually exist and NOT regimes that never shipped.
    summary = importlib.metadata.metadata("mcp-abacus")["Summary"]
    for mode in ("fixed-point", "floating-point", "rational"):
        assert mode in summary, f"Summary omits the {mode!r} mode: {summary!r}"
    # The old text claimed five regimes including two that do not exist.
    assert "fixed-width int" not in summary
    assert "decimal" not in summary


def test_keywords_name_every_headline_mode():
    # TODO 41.12: the keyword list is a discoverability surface, so it must carry all
    # three headline modes the description advertises — 'rational' was the omission.
    keywords = importlib.metadata.metadata("mcp-abacus")["Keywords"]
    for mode in ("fixed-point", "rational"):
        assert mode in keywords, f"Keywords omit {mode!r}: {keywords!r}"


def test_distribution_declares_classifiers_for_every_supported_python():
    # TODO 41.9: the distribution carries trove classifiers (audience, OS, topic, and
    # one per supported Python) for metadata completeness. The Python classifiers must
    # cover exactly 3.10-3.13, matching requires-python / the CI matrix.
    classifiers = importlib.metadata.metadata("mcp-abacus").get_all("Classifier") or []
    python_versions = {
        c.rsplit("::", 1)[1].strip()
        for c in classifiers
        if c.startswith("Programming Language :: Python :: 3.")
    }
    assert python_versions == {"3.10", "3.11", "3.12", "3.13"}
    assert any(c.startswith("Topic ::") for c in classifiers)
    assert any(c.startswith("Intended Audience ::") for c in classifiers)


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
