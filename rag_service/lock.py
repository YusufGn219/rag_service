"""A lock file so only one process updates the index at a time.

The file holds the owner's pid. A lock whose owner is dead (crash, kill) is taken over;
one whose owner is alive is respected.
"""
import contextlib
import json
import os
import time
from datetime import datetime, timezone

from rag_service.config import Config

_LOCK_NAME = "index.lock"
# A lock file that cannot be parsed may just be mid-write by its creator; only an old
# unreadable one is treated as garbage left behind.
_UNREADABLE_GRACE_SECONDS = 5.0
_STILL_ACTIVE = 259


class IndexLocked(RuntimeError):
    """Another live process holds the lock (e.g. is updating the index)."""


def lock_path(cfg: Config):
    return cfg.index_dir / _LOCK_NAME


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) would TERMINATE the process on Windows, so ask the Windows API.
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists, we just may not touch it
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_owner(path) -> tuple[int | None, str]:
    """(pid or None if unreadable, raw text)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, ""
    try:
        pid = json.loads(raw)["pid"]
        return (pid if isinstance(pid, int) else None), raw
    except (ValueError, KeyError, TypeError):
        return None, raw


def _is_stale(path) -> bool:
    pid, _ = _read_owner(path)
    if pid is not None:
        return not pid_alive(pid)
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False  # vanished: the next create attempt will sort it out
    return age > _UNREADABLE_GRACE_SECONDS


def lock_held(cfg: Config) -> bool:
    """True when a live process currently holds the lock."""
    path = lock_path(cfg)
    return path.exists() and not _is_stale(path)


def index_lock(cfg: Config):
    """Hold the index lock for the duration of the block; raise IndexLocked if someone else has it."""
    return file_lock(lock_path(cfg))


@contextlib.contextmanager
def file_lock(path):
    """Hold a pid lock file for the duration of the block; raise IndexLocked if a live process has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"pid": os.getpid(), "time": datetime.now(timezone.utc).isoformat()})
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            seen = _read_owner(path)
            if not _is_stale(path):
                owner = f"pid {seen[0]}" if seen[0] else "another process"
                raise IndexLocked(f"{path} is held by {owner}") from None
            if _read_owner(path) == seen:  # unchanged since we judged it stale: remove it
                with contextlib.suppress(OSError):
                    path.unlink()
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        break
    else:
        raise IndexLocked(f"could not take the lock {path}")
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
