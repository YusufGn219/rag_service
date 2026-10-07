"""Talk to the running HTTP service, starting it first when it is not running.

Light on purpose (no numpy, no models): the MCP bridge imports this in every Claude Code
session. Callers: bridge.py (and, later, anything else that wants the service up).
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rag_service.config import Config
from rag_service.errors import Busy, NoIndex, ServiceError, ServiceUnavailable
from rag_service.lock import IndexLocked, file_lock, pid_alive
from rag_service.version import code_version

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_TAIL_CHARS = 400


def log_path(cfg: Config) -> Path:
    return cfg.index_dir / "server.log"


def start_lock_path(cfg: Config) -> Path:
    return cfg.index_dir / "server-start.lock"


class _Pid:
    """Just enough of a Popen for ensure_running: a process we only know by pid."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        return None if pid_alive(self.pid) else 1


def _spawn_outside_job(cmd: list[str], cwd: Path, log: Path) -> _Pid:
    """Windows: start a process that is NOT a child of ours, via WMI.

    A parent may run its children in a job object that kills them when it exits (the Python
    MCP client does); a process created by the WMI service is outside that job.
    """
    inner = subprocess.list2cmdline(cmd) + f' > "{log}" 2>&1'
    command_line = f'cmd.exe /c "{inner}"'.replace("'", "''")
    script = (
        "$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
        f"@{{CommandLine='{command_line}'; CurrentDirectory='{str(cwd).replace(chr(39), chr(39) * 2)}'; "
        "ProcessStartupInformation=$si}; "
        "if ($r.ReturnValue -ne 0) { exit 1 }; $r.ProcessId"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    if out.returncode != 0 or not out.stdout.strip().isdigit():
        raise OSError(f"WMI could not start the service: {(out.stdout + out.stderr).strip()[:200]}")
    return _Pid(int(out.stdout.strip()))


def spawn_server(cfg: Config):
    """Start the service in the background: no window, and it keeps running after we exit."""
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "rag_service.server", "--port", str(cfg.port)]
    log = log_path(cfg)
    if os.name != "nt":
        with open(log, "wb") as f:
            return subprocess.Popen(cmd, cwd=_PROJECT_ROOT, start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
    try:  # first choice: a process the parent's job object cannot reach
        return _spawn_outside_job(cmd, _PROJECT_ROOT, log)
    except (OSError, subprocess.SubprocessError):
        pass
    with open(log, "wb") as f:  # fallback: detached child (survives when the parent has no kill-on-exit job)
        return subprocess.Popen(
            cmd, cwd=_PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def stop_process(pid: int) -> None:
    """Stop a process and the processes it started. A process that is already gone counts as
    stopped; raises OSError if it is still there and could not be stopped."""
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode != 0 and pid_alive(pid):
            raise OSError(f"could not stop process {pid}")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


class _Down(Exception):
    """Nothing is listening (connection refused, timeout)."""


class ServiceClient:
    def __init__(self, cfg: Config, *, spawn=spawn_server, start_timeout: float = 90.0,
                 poll_interval: float = 0.25, request_timeout: float = 300.0,
                 sleep=time.sleep, clock=time.monotonic, version=code_version, stop=stop_process):
        self._cfg = cfg
        self._spawn = spawn
        self._version = version  # the version of the code on disk right now
        self._stop = stop
        self._start_timeout = start_timeout
        self._poll = poll_interval
        self._request_timeout = request_timeout
        self._sleep = sleep
        self._clock = clock
        self._base = f"http://127.0.0.1:{cfg.port}"
        # no system proxy for a localhost call
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---- public calls (each makes sure the service is up first) ----

    def search(self, query: str, k: int = 5) -> list[dict]:
        self.ensure_running()
        return self._request("POST", "/search", {"query": query, "k": k})["results"]

    def read_note(self, path: str) -> str:
        self.ensure_running()
        return self._request("GET", "/note?" + urllib.parse.urlencode({"path": path}))["text"]

    # ---- starting ----

    def _port_open(self) -> bool:
        # On Windows a refused connection takes ~2 s to fail, so give up quickly: a running
        # service accepts on loopback at once, even while it is busy.
        try:
            with socket.create_connection(("127.0.0.1", self._cfg.port), timeout=0.3):
                return True
        except OSError:
            return False

    def _health(self) -> dict | None:
        """The service's /health answer; {} when it is up but gave none (e.g. wrong API key);
        None when it is not running."""
        if not self._port_open():
            return None
        try:
            info = self._request("GET", "/health", timeout=5.0)
        except _Down:
            return None
        except ServiceError:
            return {}  # it answered (e.g. 401): it is up, the real call will explain
        return info if isinstance(info, dict) else {}

    def healthy(self) -> bool:
        return self._health() is not None

    def _stop_if_outdated(self, info: dict) -> None:
        """A service started from older code than what is on disk now is stopped, so that the
        start below brings up the current code. Best effort: if it cannot be stopped (or is
        busy updating the index) it is simply used as it is."""
        if not info or info.get("version") == self._version():
            return
        pid = info.get("pid")
        if info.get("reindexing") or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
            return
        try:
            self._stop(pid)
        except OSError:
            return
        deadline = self._clock() + min(self._start_timeout, 10.0)
        while self._port_open() and self._clock() < deadline:
            self._sleep(self._poll)

    def ensure_running(self) -> None:
        info = self._health()
        if info:
            self._stop_if_outdated(info)
        if self.healthy():
            return
        deadline = self._clock() + self._start_timeout
        try:
            with file_lock(start_lock_path(self._cfg)):
                if self.healthy():
                    return
                try:
                    proc = self._spawn(self._cfg)
                except OSError as exc:
                    raise ServiceUnavailable(f"could not start the service: {exc}") from exc
                self._wait(proc, deadline)
        except IndexLocked:
            self._wait(None, deadline)  # someone else is starting it: just wait

    def _wait(self, proc, deadline: float) -> None:
        while not self.healthy():
            if proc is not None and proc.poll() is not None:
                raise ServiceUnavailable(f"the service exited right after starting (code {proc.poll()}). "
                                         + self._log_hint())
            if self._clock() >= deadline:
                raise ServiceUnavailable(f"the service did not come up within {self._start_timeout:.0f}s. "
                                         + self._log_hint())
            self._sleep(self._poll)

    def _log_hint(self) -> str:
        path = log_path(self._cfg)
        try:
            tail = path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:].strip()
        except OSError:
            tail = ""
        return f"Log: {path}" + (f"\n{tail}" if tail else "")

    # ---- HTTP ----

    def _request(self, method: str, path: str, body=None, *, timeout: float | None = None):
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        if self._cfg.api_key:
            headers["X-API-Key"] = self._cfg.api_key
        req = urllib.request.Request(self._base + path, data=data, method=method, headers=headers)
        try:
            with self._opener.open(req, timeout=timeout or self._request_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            _raise_for(exc.code, exc.read())
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise _Down(str(exc)) from exc


def _raise_for(code: int, raw: bytes):
    try:
        detail = json.loads(raw.decode("utf-8")).get("detail", "")
    except (ValueError, AttributeError):
        detail = raw.decode("utf-8", errors="replace")
    detail = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    if code == 401:
        raise ServiceError("the service rejected the API key (RAG_API_KEY must match on both sides)")
    if code == 409:
        raise Busy(detail)
    if code == 503:
        raise NoIndex(detail)
    raise ServiceError(detail or f"service answered HTTP {code}")
