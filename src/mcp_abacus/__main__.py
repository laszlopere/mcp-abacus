# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Console entry point: run the mcp-abacus server over stdio."""

from mcp_abacus.server import mcp


def main() -> None:
    """Start the MCP server on the default stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
