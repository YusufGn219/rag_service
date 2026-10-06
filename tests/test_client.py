import json
import socket
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from rag_service.client import ServiceClient
from rag_service.config import load_config
from rag_service.lock import file_lock
from rag_service.errors import Busy, NoIndex, ServiceError, ServiceUnavailable


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub:
    """A tiny stand-in for the real HTTP service; `start()` brings it up (like a spawned server would)."""

    def __init__(self, port, key=None):
        self.port, self.key, self.seen, self._server = port, key, [], None

    def start(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, body):
                stub.seen.append((self.command, self.path, dict(self.headers), body))
                if stub.key and self.headers.get("X-API-Key") != stub.key:
                    return self._reply(401, {"detail": "missing or wrong API key"})
                url = urlparse(self.path)
                if url.path == "/health":
                    return self._reply(200, {"loaded": False})
                if url.path == "/search":
                    code = {"noindex": 503, "busy": 409}.get(body["query"], 200)
                    if code != 200:
                        return self._reply(code, {"detail": f"stub {body['query']}"})
                    return self._reply(200, {"results": [{"query": body["query"], "k": body["k"]}]})
                if url.path == "/note":
                    path = parse_qs(url.query)["path"][0]
                    if path == "yok.md":
                        return self._reply(404, {"detail": "'yok.md' is not an indexed note"})
                    return self._reply(200, {"path": path, "text": "içerik ğüşiöç"})
                self._reply(404, {"detail": "nope"})

            def do_GET(self):
                self._handle(None)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self._handle(json.loads(self.rfile.read(n).decode("utf-8")) if n else None)

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()


class Proc:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


@pytest.fixture
def env(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    port = _free_port()
    cfg = load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx")})
    stubs = []

    def make(key=None, **kw):
        c = replace(cfg, port=port, api_key=key)
        stub = Stub(port, key)
        stubs.append(stub)
        return c, stub

    yield make
    for s in stubs:
        s.stop()


def _client(cfg, spawn, **kw):
    kw.setdefault("start_timeout", 2.0)
    kw.setdefault("poll_interval", 0.01)
    return ServiceClient(cfg, spawn=spawn, **kw)


def test_running_service_is_not_spawned(env):
    cfg, stub = env()
    stub.start()
    spawned = []
    c = _client(cfg, lambda cfg: spawned.append(1))
    assert c.search("kedi", 3) == [{"query": "kedi", "k": 3}]
    assert spawned == []


def test_down_service_is_spawned_once_and_used(env):
    cfg, stub = env()
    spawned = []

    def spawn(cfg):
        spawned.append(1)
        stub.start()
        return Proc()

    c = _client(cfg, spawn)
    assert c.search("kedi")[0]["query"] == "kedi"
    assert c.read_note("a.md") == "içerik ğüşiöç"
    assert spawned == [1]


def test_never_comes_up_gives_clear_error(env):
    cfg, _ = env()
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    (cfg.index_dir / "server.log").write_text("Traceback ... port in use", encoding="utf-8")
    c = _client(cfg, lambda cfg: Proc(), start_timeout=0.2)
    with pytest.raises(ServiceUnavailable) as exc:
        c.search("kedi")
    assert "server.log" in str(exc.value) and "port in use" in str(exc.value)


def test_spawned_process_dying_fails_fast(env):
    cfg, _ = env()
    c = _client(cfg, lambda cfg: Proc(code=1), start_timeout=30)
    with pytest.raises(ServiceUnavailable, match="exited"):
        c.search("kedi")


def test_spawn_failure_is_reported(env):
    cfg, _ = env()

    def spawn(cfg):
        raise OSError("no such file")

    with pytest.raises(ServiceUnavailable, match="no such file"):
        _client(cfg, spawn).search("kedi")


def test_another_starter_holds_the_start_lock(env):
    """If someone else is already starting the service, wait for it instead of spawning a second one."""
    cfg, stub = env()
    spawned = []
    ticks = []

    def sleep(_):
        ticks.append(1)
        if len(ticks) == 3:
            stub.start()  # the other starter's server finally comes up

    c = _client(cfg, lambda cfg: spawned.append(1), sleep=sleep)
    with file_lock(cfg.index_dir / "server-start.lock"):
        assert c.search("kedi")[0]["query"] == "kedi"
    assert spawned == []


@pytest.mark.parametrize("query,exc", [("noindex", NoIndex), ("busy", Busy)])
def test_http_errors_become_service_errors(env, query, exc):
    cfg, stub = env()
    stub.start()
    with pytest.raises(exc, match=f"stub {query}"):
        _client(cfg, None).search(query)


def test_unknown_note_is_service_error(env):
    cfg, stub = env()
    stub.start()
    with pytest.raises(ServiceError, match="not an indexed note"):
        _client(cfg, None).read_note("yok.md")


def test_api_key_is_sent(env):
    cfg, stub = env(key="s3cret")
    stub.start()
    assert _client(cfg, None).search("kedi")
    assert all({k.lower(): v for k, v in h.items()}.get("x-api-key") == "s3cret" for _, _, h, _ in stub.seen)


def test_wrong_api_key_is_explained(env):
    cfg, stub = env()
    stub.key = "other"
    stub.start()
    with pytest.raises(ServiceError, match="API key"):
        _client(cfg, None).search("kedi")


def test_turkish_text_survives_the_round_trip(env):
    cfg, stub = env()
    stub.start()
    _client(cfg, None).search("güncel durum ışık")
    assert stub.seen[-1][3]["query"] == "güncel durum ışık"


def test_pid_proxy_reports_alive_and_dead():
    import os
    import subprocess
    import sys

    from rag_service.client import _Pid

    assert _Pid(os.getpid()).poll() is None
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert _Pid(proc.pid).poll() == 1
