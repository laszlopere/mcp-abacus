# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 László Pere

"""Console entry point: run the mcp-abacus server over stdio.

The default stdio read stream is piped through `process_incoming` (TODO 43): a
tool call whose `arguments` arrive as a malformed JSON string is repaired before
the SDK's strict validation rejects it (43.1), and one that cannot be parsed at
all is answered with an actionable JSON-RPC parse error (43.2).
"""

import anyio
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage

from mcp_abacus.jsonfix import process_incoming
from mcp_abacus.server import mcp


async def _run_stdio_repaired() -> None:
    """Run the server over stdio with the argument-repair interposer (TODO 43)."""
    async with stdio_server() as (read_stream, write_stream):
        send, recv = anyio.create_memory_object_stream[SessionMessage | Exception](0)

        async def _pump() -> None:
            async with send:
                async for item in read_stream:
                    if isinstance(item, SessionMessage):
                        outcome = process_incoming(item)
                        if outcome.reply is not None:
                            await write_stream.send(outcome.reply)
                            continue
                        assert outcome.forward is not None
                        item = outcome.forward
                    await send.send(item)

        async with anyio.create_task_group() as tg, recv:
            tg.start_soon(_pump)
            await mcp._mcp_server.run(
                recv,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
            tg.cancel_scope.cancel()


def main() -> None:
    """Start the MCP server on the default stdio transport."""
    anyio.run(_run_stdio_repaired)


if __name__ == "__main__":
    main()
