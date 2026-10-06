"""MCP bridge for Claude Code:  python -m rag_service.bridge

Claude Code runs this small program per session (stdio). It offers two tools and forwards
them to the HTTP service, starting the service first if it is not running. It loads no models.

Register it (PYTHONPATH is required: Claude Code does not start us in the project folder):
  claude mcp add rag --scope user -e PYTHONPATH=<project> -- <project>/.venv/Scripts/python.exe -m rag_service.bridge
(Write the paths with / : Git Bash strips backslashes.)
"""
import sys

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rag_service.client import ServiceClient
from rag_service.config import ConfigError, load_config
from rag_service.errors import ServiceError

INSTRUCTIONS = (
    "Search the user's Obsidian notes. Use search_notes with a natural-language question "
    "(Turkish or English); it returns the best notes with their best passage. "
    "Use read_note with a result's `path` to read the whole note."
)


async def _call(fn, *args):
    """Run a (blocking) client call off the event loop; show service errors to Claude as-is."""
    try:
        return await anyio.to_thread.run_sync(fn, *args)
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc


def create_bridge(client) -> MCPServer:
    server = MCPServer("rag", instructions=INSTRUCTIONS)

    @server.tool()
    async def search_notes(query: str, k: int = 5) -> list[dict]:
        """Search the notes. Returns up to k notes (1-20), best first: path, title, date,
        score, section and text of the best passage, plus up to 2 extra passages."""
        return await _call(client.search, query, k)

    @server.tool()
    async def read_note(path: str) -> str:
        """Read one whole note. `path` is exactly the path a search result gave."""
        return await _call(client.read_note, path)

    return server


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    create_bridge(ServiceClient(cfg)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
