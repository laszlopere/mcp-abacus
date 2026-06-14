# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""mcp-abacus — type-faithful calculation MCP server."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcp-abacus")
except PackageNotFoundError:  # running from an unbuilt source tree
    __version__ = "0.0.0+unknown"
