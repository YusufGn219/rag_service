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

from rag_service import convert
from rag_service.client import ServiceClient
from rag_service.config import ConfigError, load_config
from rag_service.errors import ServiceError

INSTRUCTIONS = (
    "Search the user's Obsidian notes. Use search_notes with a natural-language question "
    "(Turkish or English); it returns the best notes with their best passage. "
    "Use read_note with a result's `path` to read the whole note, or read_section with a result's "
    "`path` and `parent.id` to read just the wider section its passage sits in."
)

CONVERT_INSTRUCTIONS = (
    " To make PDF, DOCX, XLSX, PPTX, CSV, HTML and TXT files searchable, convert them to notes: "
    "scan_files lists what is left, inspect_file checks one file, convert_file writes "
    "`<file>.md` beside it (it never overwrites anything it did not make), convert_status counts."
)


async def _call(fn, *args):
    """Run a (blocking) client call off the event loop; show service errors to Claude as-is."""
    try:
        return await anyio.to_thread.run_sync(fn, *args)
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc


def create_bridge(client, cfg=None) -> MCPServer:
    """`cfg` switches on the file-conversion tools (inspect_file, convert_file, ...)."""
    server = MCPServer("rag", instructions=INSTRUCTIONS + (CONVERT_INSTRUCTIONS if cfg else ""))

    @server.tool()
    async def search_notes(query: str, k: int = 5) -> list[dict]:
        """Search the notes. Returns up to k notes (1-20), best first: path, title, date,
        score, section and text of the best passage, plus up to 2 extra passages and `parent`
        (id, name and size in characters of the wider section the best passage sits in; null
        when that adds nothing). Read it with read_section."""
        return await _call(client.search, query, k)

    @server.tool()
    async def read_section(path: str, id: int) -> dict:
        """Read the wider section (parent) of a search result: pass its `path` and `parent.id`.
        Shorter than read_note; cut after a few thousand characters (`truncated` says so)."""
        return await _call(client.read_section, path, id)

    @server.tool()
    async def read_note(path: str) -> str:
        """Read one whole note. `path` is exactly the path a search result gave."""
        return await _call(client.read_note, path)

    if cfg is not None:
        _add_convert_tools(server, cfg)
    return server


def _add_convert_tools(server: MCPServer, cfg) -> None:
    """Tools that turn PDF/DOCX/XLSX/PPTX/CSV/HTML/TXT files into notes (see convert.py).
    They run here, in the bridge process, so converting never blocks the search service."""

    @server.tool()
    async def inspect_file(path: str) -> dict:
        """Look at a PDF/DOCX/XLSX/PPTX/CSV/HTML/TXT file without writing anything: type, size,
        pages, `hard` (little or no text came out, e.g. a scanned PDF), `modified` (last change,
        UTC), `preview` (how the text starts, about 300 characters; compare two files with it)
        and `status` (yeni, yenile, güncel or atla). `path` is absolute or relative to a vault
        root."""
        return await _call(convert.inspect_file, cfg, path)

    @server.tool()
    async def convert_file(path: str, markdown: str | None = None) -> dict:
        """Write `<file>.md` next to the file so it becomes searchable. Creates a new note or
        refreshes one rag_service made; never touches the source or a hand-written note.
        `action` says what happened (yazıldı, yenilendi, atlandı, zor). A `zor` file was not
        written: read it yourself and call again with `markdown` set to your text for it."""
        return await _call(convert.convert_file, cfg, path, markdown)

    @server.tool()
    async def scan_files(folder: str | None = None, limit: int = 200) -> dict:
        """List convertible files (under `folder`, default every vault root) with their state:
        yeni = not converted yet, yenile = source changed, güncel = up to date, atla = a
        hand-written note has that name. Files to act on come first."""
        return await _call(convert.scan, cfg, folder, limit)

    @server.tool()
    async def convert_status() -> dict:
        """Counts of convertible files over all vault roots, by state and by type."""
        return await _call(convert.status, cfg)


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    create_bridge(ServiceClient(cfg), cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
