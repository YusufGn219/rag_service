import json
import os
import subprocess
import sys
import time

import pytest

from rag_service.config import load_config
from rag_service.lock import IndexLocked, index_lock, lock_path, pid_alive


def _cfg(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    return load_config({"RAG_VAULT_ROOT": str(vault), "RAG_INDEX_DIR": str(tmp_path / "idx")})


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_acquire_and_release(tmp_path):
    cfg = _cfg(tmp_path)
    with index_lock(cfg):
        assert lock_path(cfg).is_file()
        assert json.loads(lock_path(cfg).read_text())["pid"] == os.getpid()
    assert not lock_path(cfg).exists()


def test_second_acquire_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    with index_lock(cfg):
        with pytest.raises(IndexLocked):
            with index_lock(cfg):
                pass
        assert lock_path(cfg).is_file()  # the refused attempt must not remove the holder's lock


def test_dead_owner_is_taken_over(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.index_dir.mkdir()
    lock_path(cfg).write_text(json.dumps({"pid": _dead_pid(), "time": "x"}))
    with index_lock(cfg):
        assert json.loads(lock_path(cfg).read_text())["pid"] == os.getpid()


def test_corrupt_old_lock_is_taken_over(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.index_dir.mkdir()
    lock_path(cfg).write_text("not json {")
    old = time.time() - 600
    os.utime(lock_path(cfg), (old, old))
    with index_lock(cfg):
        pass


def test_fresh_corrupt_lock_counts_as_held(tmp_path):
    """A lock file that is still being written looks corrupt for a moment; do not steal it."""
    cfg = _cfg(tmp_path)
    cfg.index_dir.mkdir()
    lock_path(cfg).write_text("")
    with pytest.raises(IndexLocked):
        with index_lock(cfg):
            pass


def test_released_when_body_fails(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(RuntimeError):
        with index_lock(cfg):
            raise RuntimeError("boom")
    assert not lock_path(cfg).exists()


def test_creates_missing_index_dir(tmp_path):
    cfg = _cfg(tmp_path)
    assert not cfg.index_dir.exists()
    with index_lock(cfg):
        assert cfg.index_dir.is_dir()


def test_pid_alive_tells_live_from_dead():
    assert pid_alive(os.getpid())
    assert not pid_alive(_dead_pid())
    assert not pid_alive(0)
