import anyio
import pytest
from mcp import Client

from rag_service.bridge import create_bridge
from rag_service.service import ServiceError


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, query, k=5):
        self.calls.append(("search", query, k))
        if query == "boom":
            raise ServiceError("there is no index yet; call reindex first")
        return [{"path": "kedi.md", "title": "kedi", "text": "kedi süt içiyor"}]

    def read_note(self, path):
        self.calls.append(("read", path))
        if path == "yok.md":
            raise ServiceError("'yok.md' is not an indexed note")
        return "# Kedi\n\nkedi süt içiyor"


def _run(coro_fn):
    async def main():
        await coro_fn()

    anyio.run(main)


def test_lists_both_tools():
    async def go():
        async with Client(create_bridge(FakeClient())) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"search_notes", "read_note"}

    _run(go)


def test_search_notes_forwards_and_returns_results():
    fake = FakeClient()

    async def go():
        async with Client(create_bridge(fake)) as c:
            res = await c.call_tool("search_notes", {"query": "kedi", "k": 3})
            assert not res.is_error
            assert "kedi süt içiyor" in str(res.content)

    _run(go)
    assert fake.calls == [("search", "kedi", 3)]


def test_search_default_k():
    fake = FakeClient()

    async def go():
        async with Client(create_bridge(fake)) as c:
            await c.call_tool("search_notes", {"query": "kedi"})

    _run(go)
    assert fake.calls == [("search", "kedi", 5)]


def test_read_note_returns_text():
    async def go():
        async with Client(create_bridge(FakeClient())) as c:
            res = await c.call_tool("read_note", {"path": "kedi.md"})
            assert not res.is_error
            assert "kedi süt içiyor" in str(res.content)

    _run(go)


@pytest.mark.parametrize("tool,args,msg", [
    ("search_notes", {"query": "boom"}, "no index"),
    ("read_note", {"path": "yok.md"}, "not an indexed note"),
])
def test_service_errors_reach_claude_as_tool_errors(tool, args, msg):
    async def go():
        async with Client(create_bridge(FakeClient())) as c:
            res = await c.call_tool(tool, args)
            assert res.is_error
            assert msg in str(res.content)

    _run(go)
