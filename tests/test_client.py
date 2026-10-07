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
        self.health = {"loaded": False, "version": "test", "pid": 4242, "reindexing": False}

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
                    return self._reply(200, stub.health)
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


def _no_stop(pid):
    pytest.fail(f"the running service must not be stopped (pid {pid})")


def _client(cfg, spawn, **kw):
    kw.setdefault("start_timeout", 2.0)
    kw.setdefault("poll_interval", 0.01)
    kw.setdefault("version", lambda: "test")  # the stub reports "test", so nothing is outdated
    kw.setdefault("stop", _no_stop)
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


# ---- a service running old code is replaced ----

def test_a_service_with_the_current_version_is_left_alone(env):
    cfg, stub = env()
    stub.health["version"] = "v1"
    stub.start()
    spawned = []
    c = _client(cfg, lambda cfg: spawned.append(1), version=lambda: "v1")
    assert c.search("kedi")[0]["query"] == "kedi"
    assert spawned == []


def _replaceable(env, *, old_version="v1", new_version="v2", **health):
    """A running stub with old code, plus the stop / spawn callables that swap it for a new one."""
    cfg, stub = env()
    stub.health.update(version=old_version, **health)
    if old_version is None:
        del stub.health["version"]
    stub.start()
    events = []

    def stop(pid):
        events.append(("stop", pid))
        stub.stop()

    def spawn(cfg):
        events.append("spawn")
        stub.health["version"] = new_version
        stub.start()
        return Proc()

    return cfg, stub, events, stop, spawn


def test_a_service_running_old_code_is_stopped_and_started_again(env):
    cfg, stub, events, stop, spawn = _replaceable(env)
    c = _client(cfg, spawn, version=lambda: "v2", stop=stop)
    assert c.search("kedi")[0]["query"] == "kedi"
    assert events == [("stop", 4242), "spawn"]
    assert stub.health["version"] == "v2"


def test_the_version_is_checked_on_every_call_but_the_new_service_is_not_restarted_again(env):
    cfg, stub, events, stop, spawn = _replaceable(env)
    c = _client(cfg, spawn, version=lambda: "v2", stop=stop)
    c.search("kedi")
    c.search("araba")
    c.read_note("a.md")
    assert events == [("stop", 4242), "spawn"]


def test_a_service_that_reports_no_version_counts_as_old(env):
    cfg, stub, events, stop, spawn = _replaceable(env, old_version=None)
    assert _client(cfg, spawn, version=lambda: "v2", stop=stop).search("kedi")
    assert events == [("stop", 4242), "spawn"]


def test_a_service_that_is_updating_the_index_is_not_interrupted(env):
    cfg, stub, events, stop, spawn = _replaceable(env, reindexing=True)
    c = _client(cfg, spawn, version=lambda: "v2", stop=stop)
    assert c.search("kedi")[0]["query"] == "kedi"
    assert events == []


def test_if_the_old_service_cannot_be_stopped_it_is_still_used(env):
    cfg, stub, events, _, spawn = _replaceable(env)

    def stop(pid):
        raise OSError("access denied")

    assert _client(cfg, spawn, version=lambda: "v2", stop=stop).search("kedi")
    assert events == []


def test_a_service_without_a_pid_is_not_stopped(env):
    cfg, stub, events, stop, spawn = _replaceable(env)
    del stub.health["pid"]
    assert _client(cfg, spawn, version=lambda: "v2", stop=stop).search("kedi")
    assert events == []


def test_a_wrong_api_key_never_triggers_a_restart(env):
    cfg, stub, events, stop, spawn = _replaceable(env)
    stub.key = "other"
    with pytest.raises(ServiceError, match="API key"):
        _client(cfg, spawn, version=lambda: "v2", stop=stop).search("kedi")
    assert events == []


def test_a_service_that_does_not_go_away_is_still_used(env):
    cfg, stub, events, _, spawn = _replaceable(env)
    c = _client(cfg, spawn, version=lambda: "v2", stop=lambda pid: events.append(("stop", pid)),
                start_timeout=0.2)
    assert c.search("kedi")  # stop "worked" but the port stays open: keep using the old one
    assert events == [("stop", 4242)]


def test_stop_process_ends_a_running_process_and_accepts_one_that_is_gone():
    import subprocess
    import sys

    from rag_service.client import stop_process

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stop_process(proc.pid)
    assert proc.wait(timeout=10) != 0
    stop_process(proc.pid)  # already gone: no error


def test_pid_proxy_reports_alive_and_dead():
    import os
    import subprocess
    import sys

    from rag_service.client import _Pid

    assert _Pid(os.getpid()).poll() is None
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert _Pid(proc.pid).poll() == 1
